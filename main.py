import argparse
import random
import os
import time
import sys
import pickle

import utils
import oversamplers
import undersamplers
import thresholds
import losses
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
                        choices=['RUS', 'NCR', 'IHT', 'ENN', 'TomekLinks', 'IPF'],
                        help="Which undersampling strategy should we choose?")
    parser.add_argument('-hybrid', default = None,
                        choices=["majority", "all"],
                        help="Which hybrid oversampling/undersammpling strategy should we choose?")
    parser.add_argument('-label_smoothing', default = False,
                        action="store_true",
                        help="Should we perform label smoothing?")
    parser.add_argument('-clf', default = "base",
                        help="Which classifier should we use?")
    parser.add_argument('-loss', default = None,
                        choices = ["CrossEntropy", "FocalLoss", "GradientHarmonized"],
                        help="Which loss function should we use?")
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
    clf_steps = []
    param_grid = {}
    prep_param_grid = {}
    c_time = None
    configs = []

    if opt.oversampling is not None and opt.undersampling is not None and opt.hybrid is None:
       opt.hybrid = "majority"
    
    if opt.oversampling is not None:
        oversampler= oversamplers.fetch_oversampler(opt.oversampling)
        steps = steps + [("oversampler", oversampler)]

        if opt.hybrid is not None and os.path.exists(f"config/{opt.oversampling}.pkl") \
                                  and os.path.exists(f"info/{opt.oversampling}.pkl"):
            print(f"Fetching oversamplier model best configuration...")
            with open(f"config/{opt.oversampling}.pkl", "rb") as fp:
                oversampler_config = pickle.load(fp)
            print(f"Config: {oversampler_config}")
            
            with open(f"info/{opt.oversampling}.pkl", "rb") as fp:
                oversampler_info = pickle.load(fp)
                print(f"Adding {oversampler_info['time']} to training time")
                c_time = oversampler_info["time"]

            configs.append(oversampler_config)
        else:
            if oversampler.parameter_grid() is not None:
                for (parameter, values) in oversampler.parameter_grid().items():
                    prep_param_grid["oversampler__" + str(parameter)] = values

        name += f"_{opt.oversampling}"
        if name.startswith("Base"):
            name = name[5:]
        oversampler.set_params(**{"categorical_features":len(cat_feats)})

    if opt.undersampling is not None:
        undersampler = undersamplers.fetch_undersampler(opt.undersampling)
        steps = steps + [("undersampler", undersampler)]

        if undersampler.parameter_grid() is not None:
            for (parameter, values) in undersampler.parameter_grid().items():
                prep_param_grid["undersampler__" + str(parameter)] = values
        name += f"_{opt.undersampling}"
        if name.startswith("Base"):
            name = name[5:]
        undersampler.set_params(**{"categorical_features":len(cat_feats)})

    if opt.hybrid is not None and (opt.oversampling is None or opt.oversampling is None):
        print("Can only use hybrid sampling if both oversampling or undersampling strategy are defined")
        sys.exit(-1)
    
    if opt.hybrid is not None:
        undersampler.set_params(**{"cls":opt.hybrid})
        if opt.hybrid == "all":
            name += "_all"
       
    clf = LGBM()

    if opt.loss is not None:
        loss = losses.fetch_loss(opt.loss)
        name += f"_{opt.loss}"
        if name.startswith("Base"):
            name = name[5:]
        clf.set_params(**{"loss_fn": loss})
        clf.untoggle_param_grid("loss")
    else:
        loss = LGBM().loss_fn
    
    if opt.label_smoothing:
        name += "_LS"
        if name.startswith("Base"):
            name = name[5:]
        loss.set_params(**{"smooth": True})

    steps = steps + [("clf", clf)]
    if name == "Base":
        clf.untoggle_param_grid("clf")
    clf_steps = clf_steps + [("clf", clf)]
    for (parameter, values) in clf.parameter_grid().items():
        param_grid["clf__" + str(parameter)] = values
    clf.set_params(**{"categorical_features":len(cat_feats)})
    
    pipeline = Pipeline(steps=steps)
    clf_pipeline = Pipeline(steps = clf_steps)

    start_time = time.time()
    if c_time is not None:
        start_time -= c_time
    for config in configs:
        pipeline.set_params(**config)
    
    print(f"Here is my grid: {prep_param_grid}")

    # To do hyperparameter search step by step we must first have a defined classifier
    if name != "Base" and os.path.exists("config/Base.pkl") and os.path.exists("info/Base.pkl"):
        print(f"Fetching base model best configuration...")
        with open("config/Base.pkl", "rb") as fp:
            base_config = pickle.load(fp)
        
        with open("info/Base.pkl", "rb") as fp:
            base_info = pickle.load(fp)
            print(f"Adding {base_info['time']} to training time")
            start_time -= base_info["time"]
        #clf.set_params(**base_config)
        clf_pipeline.set_params(**base_config)
        pipeline.set_params(**base_config)
    elif name != "Base":
        print(f"Base model best configuration not found. Training base model...")
        search_clf = CustomCV(estimator = pipeline, 
                      param_distributions = param_grid,
                      scoring = make_scorer(auc, greater_is_better=True, needs_proba = True), 
                      n_jobs=1, verbose = 2, cv = 2, error_score = 0,
                      n_iter = 100, random_state = 42)
        clf = search_clf.fit(X_train, y_train)
    
    if "undersampler" in pipeline.named_steps:
        pipeline.named_steps["undersampler"].set_params(**{"estimator": clf})

    search = CustomCV(clf = clf_pipeline,
                      estimator = pipeline, 
                      param_distributions = param_grid, 
                      n_iter = 20 * len(param_grid.keys()),
                      prep_param_distributions = prep_param_grid,
                      prep_n_iter = 20 * len(prep_param_grid.keys()),
                      scoring = make_scorer(auc, greater_is_better=True, needs_proba = True), 
                      n_jobs=1, verbose = 2, cv = 2, error_score = 0, random_state = 42)
                      
    est = search.fit(X_train, y_train)
    end_time = time.time()

    test_start = time.time()
    probs = est.predict_proba(X_test)[:, 1]
    train_probs = est.predict_proba(X_train)[:, 1]

    th = search.th#thresholds.fetch_threshold(opt.threshold, y_train, train_probs)
    y_pred = est.predict(X_test, **{"th":th})
    test_end = time.time()
    test_time = test_end - test_start

    total_time = end_time - start_time
    print(f'Ended training and classification in {round(total_time + test_time, 2)} seconds.\n')
    
    # TODO: add more conditions in the future
    if opt.loss is None:
        metric_file = "preprocessing"
    
        subtype = "Base"
        if opt.oversampling and not opt.undersampling:
            subtype = "Oversamplers"
        
        if not opt.oversampling and opt.undersampling:
            subtype = "Undersamplers"

        if opt.oversampling and opt.undersampling:
            subtype = "Hybrid"
        
        if opt.label_smoothing:
            subtype = "Label Smoothing"
    else:
        metric_file = "inprocessing"

        # TODO: add more subtypes in the future 
        subtype = "Losses"

    utils.dump_metrics(y_test, y_pred, probs, name, total_time, test_time, metric_file, subtype, search.best_params_)

    if opt.plot_scores:
        positives = probs[y_test == 1]
        negatives = probs[y_test == 0]
        plt.hist(probs[y_test == 1], bins = 100, range = (0, 1), label = "Positive", alpha = 0.5, weights = np.ones_like(positives)/float(len(positives)))
        plt.hist(probs[y_test == 0], bins = 100, range = (0, 1), label = "Negative", alpha = 0.5, weights = np.ones_like(negatives)/float(len(negatives)))
        plt.axvline(x = th, alpha = 0.5, color = "red", linestyle = "--", linewidth = 1)
        plt.legend(loc='upper right')
        results_dir = f'results/{metric_file}/{name}/'
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'{name}_probs.pdf'))

if __name__ == '__main__':
    main()