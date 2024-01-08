import argparse
import random
import os
import time
import sys

import utils
import oversamplers
import undersamplers
import hybrid
import meta_learners
import thresholds
from losses import LabelSmoothingLoss, FocalLoss
from classifiers import LGBM
from classifiers import CustomRandomForest
from CustomCV import CustomCV

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


def auc(y_test, y_pred):    
    return sklearn.metrics.roc_auc_score(y_test, y_pred)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        choices=['baf'],
                        help="Which dataset should we choose?")
    parser.add_argument('-ensemble', default = None,
                        help="What is the ensemble strategy?")
    parser.add_argument('-data_subsampling', default = 1.00,
                        type = float,
                        help="By how much should the dataset be reduced?")
    parser.add_argument('-classifier', default = 'Base',
                        choices=['Base', 'RF'],
                        help="Which classifier should we choose?")
    parser.add_argument('-oversampling', default = None,
                        choices=['SMOTE', 'ADASYN', 'KMeansSMOTE', 'RACOG'],
                        help="Which oversampling strategy should we choose?")
    parser.add_argument('-undersampling', default = None,
                        choices=['RUS', 'TomekLinks', 'ENN'],
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
    parser.add_argument('-threshold', default = 'ROC',
                        choices=['CS', 'ROC'],
                        help="Which threshold-moving strategy should we choose?")
    
    return parser.parse_args()


def generate_subsets(X_train, y_train, subsampler_name, j=2):
    subsampler = None
    if subsampler_name == 'RUS':
        subsampler = undersamplers.RUSWrapper(random_state=42)
    if subsampler_name == 'ENN':
        subsampler = undersamplers.ENNWrapper(random_state=42)
    if subsampler_name == 'TL':
        subsampler = undersamplers.TomekLinksWrapper(random_state=42)

    subsets = []
    out_of_subsets = []

    for _ in range(j):
        subsampler.random_state = np.random.randint(1, 1000)

        subsampler.fit_resample(X_train, y_train)

        all_indices = list(range(len(X_train)))
        sub_indices = np.sort(subsampler.sample_indices_)
        oos_indices = np.array([i for i in all_indices if i not in sub_indices])

        X_resampled, y_resampled = X_train.iloc[sub_indices], y_train.iloc[sub_indices]
        X_out_of_subset, y_out_of_subset = X_train.iloc[oos_indices], y_train.iloc[oos_indices]

        subsets.append((X_resampled, y_resampled))
        out_of_subsets.append((X_out_of_subset, y_out_of_subset))

    return subsets, out_of_subsets


def preprocess(opt, cat_feats, name):
    steps = []
    param_grid = {}

    if opt.label_smoothing:
        name += "_LS"

    if opt.oversampling is not None and opt.undersampling is not None:
        print("Cannot perform oversampling and undersampling approaches simulaneously! Check hybrid option instead.")
        sys.exit(-1)
    
    if opt.hybrid is not None and opt.oversampling is None:
        print("Can only use hybrid sampling if oversampling strategy is defined")
        sys.exit(-1)
    
    if opt.oversampling is not None and opt.hybrid is None:
        preprocessor = oversamplers.fetch_oversampler(opt.oversampling)
        steps = steps + [("preprocessor", preprocessor)]

        if preprocessor.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                param_grid["preprocessor__" + str(parameter)] = values

        name += f"_{opt.oversampling}"
        preprocessor.set_params(**{"categorical_features":len(cat_feats)})

    if opt.undersampling is not None:
        preprocessor = undersamplers.fetch_undersampler(opt.undersampling)
        steps = steps + [("preprocessor", preprocessor)]

        if preprocessor.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                param_grid["preprocessor__" + str(parameter)] = values
        name += f"_{opt.undersampling}"
        preprocessor.set_params(**{"categorical_features":len(cat_feats)})

    
    if opt.hybrid is not None:
        preprocessor = hybrid.fetch_hybrid(opt.hybrid)
        steps = steps + [("preprocessor", preprocessor)]

        if preprocessor.parameter_grid() is not None:
            for (parameter, values) in preprocessor.parameter_grid().items():
                param_grid["preprocessor__" + str(parameter)] = values
        name += f"_{opt.undersampling}"
        preprocessor.set_params(**{"categorical_features":len(cat_feats)})

    return (param_grid, steps, name)


def clf_LGBM(opt, name):
    if opt.label_smoothing:
        name += "_LS"
        ls = LabelSmoothingLoss()
        clf = LGBM(ls)

    else:
        ls = LabelSmoothingLoss(0, freeze = True)
        clf = LGBM(ls)
    
    return (clf, name)


def clf_RF(opt, name):
    if opt.label_smoothing:
        name += "_LS"
        ls = LabelSmoothingLoss()
        clf = CustomRandomForest(ls)

    else:
        ls = LabelSmoothingLoss(0, freeze = True)
        clf = CustomRandomForest(ls)

    return (clf, name)


def train(opt, param_grid, steps, cat_feats, X_train, y_train, name):
    if opt.classifier == 'Base':
        clf, name = clf_LGBM(opt, name)
    elif opt.classifier == 'RF':
        clf, name = clf_RF(opt, name)
    else:
        print("Wrong classifier name!")
        sys.exit(-1)

    steps = steps + [("clf", clf)]
    for (parameter, values) in clf.parameter_grid().items():
        param_grid["clf__" + str(parameter)] = values
    pipeline = Pipeline(steps=steps)
    if "preprocessor" in pipeline.named_steps:
        prep_name = "preprocessor"
    else:
        prep_name = None

    print(param_grid)

    is_oversampler = opt.oversampling is not None
    is_undersampler = opt.undersampling is not None
    search = CustomCV(prep_name = prep_name, estimator = pipeline, param_distributions = param_grid,
                      scoring = make_scorer(auc, greater_is_better=True, needs_proba = True), 
                      n_jobs=1, verbose = 2, error_score = "raise", refit = False, cv = 2,
                      oversampler = is_oversampler, undersampler = is_undersampler, n_iter = 10, random_state = 42)
    
    search.fit(X_train, y_train, generate_metadata = opt.ensemble is not None, clf__categorical_features = len(cat_feats))

    print("CV Preds:")
    print(search.cv_predictions)

    return search


def classify(opt, search, X_train, y_train, X_test):
    probs = search.predict_proba(X_test)[:, 1]
    train_probs = search.predict_proba(X_train)[:, 1]

    th = thresholds.fetch_threshold(opt.threshold, y_train, train_probs)
    y_pred = search.best_estimator_.predict(X_test, **{"th":th})

    return (y_pred, probs, th)


def main():
    opt = parse_arguments()

    name = opt.classifier

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

    num_iterations = 1
    subsets = [(X_train, y_train)]
    out_of_subsets = None

    classifiers = []
    metadata_matrix = np.empty((len(X_train),))

    start_time = time.time()
    print('Starting training...')
    
    if opt.ensemble is not None:
        num_iterations = int(opt.ensemble.split('-')[1])
        subsets, out_of_subsets = generate_subsets(X_train, y_train, opt.ensemble.split('-')[0], j=num_iterations)

    for i in range(num_iterations):
        param_grid, steps, name = preprocess(opt, cat_feats, name)

        search = train(opt, param_grid, steps, cat_feats, subsets[i][0], subsets[i][1], name)
        classifiers.append(search)

        if opt.ensemble is not None:
            y_pred, _, _ = classify(opt, search, subsets[i][0], subsets[i][1], out_of_subsets[i][0])
            pred = np.hstack((np.array(search.cv_predictions), y_pred))

            metadata_matrix = np.vstack((metadata_matrix, np.array(pred)))

        # if opt.ensemble is not None:
        #     # TODO: RFs
        #     opt.classifier = 'RF'
        #     search = train(opt, param_grid, steps, cat_feats, X_train, y_train, name)
        #     classifiers.append(search)
        #     metadata_matrix = np.vstack((metadata_matrix, np.array(search.cv_predictions)))
        #     opt.classifier = 'Base'


    if opt.ensemble is not None:
        metadata_matrix = metadata_matrix[1:].T
        
        print("Metadata Shape:")
        print(metadata_matrix.shape)

        # Train meta-learner
        meta_learner = meta_learners.fetch_metalearner(opt.ensemble.split('-')[2])

        meta_learner.fit(metadata_matrix, y_train)
        

    predictions = np.empty((len(X_test),))
    probs_list = []
    th = 0
    for i in range(num_iterations):
        y_pred, probs, th = classify(opt, classifiers[i], X_train, y_train, X_test)
        predictions = np.vstack((predictions, np.array(y_pred)))
        probs_list.append(probs)

        # if opt.ensemble is not None:
        #     y_pred, probs, th = classify(opt, classifiers[2*i+1], X_train, y_train, X_test)
        #     predictions = np.vstack((predictions, np.array(y_pred)))

    if opt.ensemble is not None:
        name = "US-SE_" + opt.ensemble + "_" + name
        y_pred = meta_learner.predict(predictions[1:].T)
        probs = meta_learner.predict_proba(predictions[1:].T)[:, 1]
        print(y_pred[y_test == 0])
        # th = ?
    else:
        y_pred = predictions[1]
        probs = probs_list[0]

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Ended training and classification in {round(total_time, 2)} seconds.\n')

    utils.dump_metrics(y_test, y_pred, probs, name, total_time)

    if opt.plot_scores:
        positives = probs[y_test == 1]
        negatives = probs[y_test == 0]
        plt.hist(probs[y_test == 1], bins = 100, range = (0, 1), label = "Positive", alpha = 0.5, weights = np.ones_like(positives)/float(len(positives)))
        plt.hist(probs[y_test == 0], bins = 100, range = (0, 1), label = "Negative", alpha = 0.5, weights = np.ones_like(negatives)/float(len(negatives)))
        plt.axvline(x = th, alpha = 0.5, color = "red", linestyle = "--", linewidth = 1)
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    main()