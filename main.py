import argparse
import random
import os
import time
import sys

import utils
import oversamplers
import undersamplers
from losses import LabelSmoothingLoss, FocalLoss
from classifiers import LGBM

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.model_selection
from sklearn import metrics

import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import sklearn.metrics


def f1_measure(y_test, y_pred):
    TP = np.sum(y_pred[y_test == 1.0])
    TN = np.sum(1 - y_pred[y_test == 0])
    FP = np.sum(y_pred[y_test == 0])
    FN = np.sum(1 - y_pred[y_test == 1.0])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision == 0 or recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)

def auc(y_test, y_pred):    
    return sklearn.metrics.roc_auc_score(y_test, y_pred)

def compute_cutoff(y_test, y_pred):
    fpr, tpr, th = sklearn.metrics.roc_curve(y_test, y_pred)
    J_statistic = tpr + 1 - fpr
    return th[np.argmax(J_statistic)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        choices=['baf'],
                        help="Which dataset should we choose?")
    parser.add_argument('-oversampling', default = None,
                        choices=['SMOTE', 'SMOTENC', 'ADASYN', 'KMeansSMOTE'],
                        help="Which oversampling strategy should we choose?")
    parser.add_argument('-undersampling', default = None,
                        choices=['RUS'],
                        help="Which undersampling strategy should we choose?")
    parser.add_argument('-hybrid', default = None,
                        choices=['TODO'],
                        help="Which hybrid oversampling/undersammpling strategy should we choose?")
    parser.add_argument('-label_smoothing', default = False,
                        action="store_true",
                        help="Should we perform label smoothing?")
    parser.add_argument('-plot_scores', default = False,
                        action="store_true",
                        help="Should we perform label smoothing?")
    
    opt = parser.parse_args()

    name = "Base"

    data = utils.fetch_data(opt.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    #sys.exit()

    cat_feats = data["cat_feats"]

    steps = []
    param_grid = {}

    if opt.label_smoothing:
        name += "_LS"

    if opt.oversampling is not None and opt.undersampling is not None:
        print("Cannot perform oversampling and undersampling approaches simulaneously! Check hybrid option instead.")
        sys.exit(-1)
    
    if opt.hybrid is not None and opt.undersampling is not None:
        print("Can only use hybrid sampling if oversampling strategy is defined")
        sys.exit(-1)
    
    if opt.oversampling is not None:
        print("Will perform oversampling!")
        preprocessor = oversamplers.fetch_oversampler(opt.oversampling)
        steps = steps + [("preprocessor", preprocessor)]

        if preprocessor.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                param_grid["preprocessor__" + str(parameter)] = values

        name += f"_{opt.oversampling}"

    if opt.undersampling is not None:
        print("Will perform undersampling!")
        preprocessor = undersamplers.fetch_undersampler(opt.undersampling)
        steps = steps + [("preprocessor", preprocessor)]

        if preprocessor.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                param_grid["preprocessor__" + str(parameter)] = values
        name += f"_{opt.undersampling}"

    if opt.label_smoothing:
        name += "_LS"
        ls = LabelSmoothingLoss()
        clf = LGBM(ls)

    else:
        ls = LabelSmoothingLoss(0, freeze = True)
        clf = LGBM(ls)
    #preprocessor.set_params(**{"categorical_features":cat_feats})
    #print(getattr(preprocessor, "categorical_features"))
    steps = steps + [("clf", clf)]
    for (parameter, values) in clf.parameter_grid().items():
        param_grid["clf__" + str(parameter)] = values
    pipeline = Pipeline(steps=steps)
    start_time = time.time()
    print('Starting training...')
    search = GridSearchCV(pipeline, param_grid, scoring = make_scorer(auc, greater_is_better=True, needs_proba = True), 
                            n_jobs=1, verbose = 4, error_score = "raise", refit = True, cv = 2)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    probs = search.predict_proba(X_test)[:, 1]
    test_probs = search.predict_proba(X_train)[:, 1]

    th = compute_cutoff(y_train, test_probs)
    y_pred = search.best_estimator_.predict(X_test, **{"th":th})

    #y_pred = pipeline["clf"].predict(X_test)
    #probs = pipeline["clf"].predict_proba(X_test)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Ended training and classification in {round(end_time - start_time, 2)} seconds.\n')

    #display = sklearn.metrics.RocCurveDisplay.from_predictions(y_test, probs)
    #plt.show()
    utils.dump_metrics(y_test, y_pred, probs, name, total_time)

    if opt.plot_scores:
        #probs = search.best_estimator_["clf"].predict_proba(X_test)
        #probs = pipeline["clf"].predict_proba(X_test)[:, 1]
        positives = probs[y_test == 1]
        negatives = probs[y_test == 0]
        plt.hist(probs[y_test == 1], bins = 100, range = (0, 1), label = "Positive", alpha = 0.5, weights = np.ones_like(positives)/float(len(positives)))
        plt.hist(probs[y_test == 0], bins = 100, range = (0, 1), label = "Negative", alpha = 0.5, weights = np.ones_like(negatives)/float(len(negatives)))
        plt.axvline(x = th, alpha = 0.5, color = "red", linestyle = "--", linewidth = 1)
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    main()