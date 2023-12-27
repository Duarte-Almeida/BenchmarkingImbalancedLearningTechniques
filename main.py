import argparse
import random
import os
import time
import sys
import pickle

import utils
import oversamplers
import undersamplers
import hybrid
import thresholds
from losses import LabelSmoothingLoss, FocalLoss
from classifiers import LGBM
from CustomCV import CustomCV

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.model_selection
from sklearn import metrics

import matplotlib.pyplot as plt


def auc(y_test, y_pred):    
    return sklearn.metrics.roc_auc_score(y_test, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        choices=['baf'],
                        help="Which dataset should we choose?")
    parser.add_argument('-data_subsampling', default = 1.00,
                        type = float,
                        help="By how much should the dataset be reduced?")
    parser.add_argument('-oversampling', default = None,
                        choices=['SMOTE', 'ADASYN', 'KMeansSMOTE', 'RACOG'],
                        help="Which oversampling strategy should we choose?")
    parser.add_argument('-undersampling', default = None,
                        choices=['RUS', 'NCR', 'InstanceHardness'],
                        help="Which undersampling strategy should we choose?")
    parser.add_argument('-hybrid', default = None,
                        choices=['IPF'],
                        help="Which hybrid oversampling/undersammpling strategy should we choose?")
    parser.add_argument('-label_smoothing', default = False,
                        action="store_true",
                        help="Should we perform label smoothing?")
    parser.add_argument('-plot_scores', default = False,
                        action="store_true",
                        help="Should we perform label smoothing?")
    parser.add_argument('-threshold', default = 'FPR',
                        choices=['CS', 'J', 'FPR'],
                        help="Which threshold-moving strategy should we choose?")
    
    opt = parser.parse_args()

    name = "Base"

    data = utils.fetch_data(opt.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    cat_feats = data["cat_feats"]

    np.random.seed(42)
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    if opt.data_subsampling < 1.00:
        X_train = X_train.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        sub_idx = np.random.choice(X_train.shape[0], size = int(X_train.shape[0] * opt.data_subsampling), replace = False)
        X_train, y_train = X_train.loc[sub_idx], y_train.loc[sub_idx]

    steps = []
    param_grid = {}
    over_param_grid = {}
    under_param_grid = {}


    #if opt.oversampling is not None and opt.undersampling is not None:
    #    print("Cannot perform oversampling and undersampling approaches simulaneously! Check hybrid option instead.")
    #    sys.exit(-1)
    
    if opt.hybrid is not None and opt.oversampling is None:
        print("Can only use hybrid sampling if oversampling strategy is defined")
        sys.exit(-1)
    
    if opt.oversampling is not None and opt.hybrid is None:
        oversampler= oversamplers.fetch_oversampler(opt.oversampling)
        steps = steps + [("oversampler", preprocessor)]

        if oversampler.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                over_param_grid["oversampler__" + str(parameter)] = values

        name += f"_{opt.oversampling}"
        oversampler.set_params(**{"categorical_features":len(cat_feats)})

    if opt.undersampling is not None:
        undersampler = undersamplers.fetch_undersampler(opt.undersampling)
        steps = steps + [("undersampler", undersampler)]

        if undersampler.parameter_grid() is not None:
            for (parameter, values) in undersampler.parameter_grid().items():
                under_param_grid["undersampler__" + str(parameter)] = values
        name += f"_{opt.undersampling}"
        undersampler.set_params(**{"categorical_features":len(cat_feats)})

    
    if opt.hybrid is not None:
        preprocessor = hybrid.fetch_hybrid(opt.hybrid)
        steps = steps + [("preprocessor", preprocessor)]

        if preprocessor.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                prep_param_grid["preprocessor__" + str(parameter)] = values
        name += f"_{opt.undersampling}"
        name += f"_{opt.oversampling}" + f"_{opt.hybrid}"
        preprocessor.set_params(**{"categorical_features":len(cat_feats)})


    if opt.label_smoothing:
        name += "_LS"
        ls = LabelSmoothingLoss()
        clf = LGBM(ls)

    else:
        ls = LabelSmoothingLoss(0, freeze = True)
        clf = LGBM(ls)

    steps = steps + [("clf", clf)]
    for (parameter, values) in clf.parameter_grid().items():
        param_grid["clf__" + str(parameter)] = values
    clf.set_params(**{"categorical_features":len(cat_feats)})
    

    pipeline = Pipeline(steps=steps)

    start_time = time.time()
    print('Starting training...')

    # To do hyperparameter search step by step we must first have a defined classifier
    if name != "Base" and os.path.exists("config/Base.pkl"):
        print(f"Fetching base model best configuration...")
        with open("config/Base.pkl", "rb") as fp:
            base_config = pickle.load(fp)
        clf.set_params(**base_config)
    elif name != "Base":
        print(f"Base model best configuration not found. Training base model...")
        search_clf = CustomCV(estimator = pipeline, 
                      param_distributions = param_grid,
                      scoring = make_scorer(auc, greater_is_better=True, needs_proba = True), 
                      n_jobs=1, verbose = 2, cv = 2, error_score = 0,
                      n_iter = 100, random_state = 42)
        clf = search_clf.fit(X, y)
    
    if "undersampler" in pipeline.named_steps:
        pipeline.named_steps["undersampler"].set_params(**{"estimator": clf})

    search = CustomCV(estimator = pipeline, 
                      param_distributions = param_grid, 
                      n_iter = 50,
                      prep_param_distributions = {
                        "oversampler": over_param_grid, 
                        "undersampler": under_param_grid
                      },
                      prep_n_iter = [10, 10],
                      scoring = make_scorer(auc, greater_is_better=True, needs_proba = True), 
                      n_jobs=1, verbose = 2, cv = 2, error_score = 0, random_state = 42)
                      
    est = search.fit(X_train, y_train)

    probs = est.predict_proba(X_test)[:, 1]
    train_probs = est.predict_proba(X_train)[:, 1]

    th = thresholds.fetch_threshold(opt.threshold, y_train, train_probs)
    y_pred = est.predict(X_test, **{"th":th})

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Ended training and classification in {round(end_time - start_time, 2)} seconds.\n')

    utils.dump_metrics(y_test, y_pred, probs, name, total_time, search.best_params_)

    if opt.plot_scores:
        positives = probs[y_test == 1]
        negatives = probs[y_test == 0]
        plt.hist(probs[y_test == 1], bins = 100, range = (0, 1), label = "Positive", alpha = 0.5, weights = np.ones_like(positives)/float(len(positives)))
        plt.hist(probs[y_test == 0], bins = 100, range = (0, 1), label = "Negative", alpha = 0.5, weights = np.ones_like(negatives)/float(len(negatives)))
        plt.axvline(x = th, alpha = 0.5, color = "red", linestyle = "--", linewidth = 1)
        plt.legend(loc='upper right')
        results_dir = f'results/{name}/'
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'{name}_probs.pdf'))
        plt.show()

if __name__ == '__main__':
    main()