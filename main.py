import argparse
import random
import os
import time
import sys
import pickle

import utils
import oversamplers
import undersamplers
import losses
import classifiers
import ensembles
from CustomCV import CustomCV

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
import sklearn.metrics
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.model_selection
from sklearn import metrics
import sklearn.utils

import matplotlib.pyplot as plt


def auc(y_test, y_pred):    
    return sklearn.metrics.roc_auc_score(y_test, y_pred)

def compute_metrics(estimator, X_train, y_train, X_test, y_test):
    aucs = []
    tprs = []
    start = time.time()
    estimator.fit(X_train, y_train)
    end = time.time()
    probs = estimator.predict_proba(X_test)[:, 1]
    n_iterations = 100

   
    for _ in range(n_iterations):
        # Create a bootstrap sample
        y_test_bootstrap, y_prob_bootstrap = sklearn.utils.resample(y_test, probs)
        
        fpr, tpr, th = sklearn.metrics.roc_curve( y_test_bootstrap, y_prob_bootstrap)
        lower_fpr_idx = np.argmax(fpr[fpr < 0.05])
        tpr_score = tpr[lower_fpr_idx]
        auc = sklearn.metrics.roc_auc_score( y_test_bootstrap, y_prob_bootstrap)

        aucs.append(auc)
        tprs.append(tpr_score)
        
    return {
        "AUC": np.mean(aucs),
        "AUC_std": np.std(aucs),
        "TPR": np.mean(tprs),
        "TPR_std": np.std(tprs),
        "Time": start - end,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default = None,
                         choices=['baf', 'mlg'],
                        help="Which dataset should we choose?")
    parser.add_argument('-data_subsampling', default = 0.1,
                        type = float,
                        help="By how much should the dataset be reduced?")
    parser.add_argument('-oversampling', default = None,
                        choices=['SMOTE', 'ADASYN', 'KMeansSMOTE', 'RACOG'],
                        help="Which oversampling strategy should we choose?")
    parser.add_argument('-undersampling', default = None,
                        choices=['RUS', 'NCR', 'IHT'],
                        help="Which undersampling strategy should we choose?")
    parser.add_argument('-clf', default = "Base",
                        choices = ["Base"],
                        help="Which classifier should we use?")
    parser.add_argument('-loss', default = None,
                        choices = ["WeightedCrossEntropy", "LabelSmoothing", "LabelRelaxation", 
                                   "FocalLoss", "GradientHarmonized"],
                        help="Which loss function should we use?")
    parser.add_argument('-plot_scores', default = False,
                        action="store_true",
                        help="Should we perform label smoothing?")
    parser.add_argument('-simplified', action='store_true',
                        help="Should we consider a simplified base model")
    parser.add_argument('-ensemble', default = None,
                        choices=['StackedEnsemble', "SelfPaced"],
                        help="Which ensembling strategy should we apply?")
    parser.add_argument('-n_iter', default = 50, type = int)

    opt = parser.parse_args()

    name = "Base"

    data = utils.fetch_data(opt.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    cat_feats = data["cat_feats"]

    np.random.seed(42)
    if opt.data_subsampling < 1.00:
        sub_idx = np.random.choice(X_train.shape[0], size = int(X_train.shape[0] * opt.data_subsampling), replace = False)
        X_train = X_train[sub_idx]
        y_train = y_train[sub_idx]

    steps = []
    param_grid = {}

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
            with open(f"config/{opt.dataset}/{opt.oversampling}.pkl", "rb") as fp:
                oversampler_config = pickle.load(fp)
            print(f"Config: {oversampler_config}")
            
            with open(f"info/{opt.dataset}/{opt.oversampling}.pkl", "rb") as fp:
                oversampler_info = pickle.load(fp)
                print(f"Adding {oversampler_info['time']} to training time")
                c_time = oversampler_info["time"]

            configs.append(oversampler_config)
        else:
            if oversampler.parameter_grid() is not None:
                for (parameter, values) in oversampler.parameter_grid().items():
                    param_grid["oversampler__" + str(parameter)] = values

        name += f"_{opt.oversampling}"
        if name.startswith("Base"):
            name = name[5:]
        oversampler.set_params(**{"categorical_features":len(cat_feats)})

    if opt.undersampling is not None:
        undersampler = undersamplers.fetch_undersampler(opt.undersampling)
        steps = steps + [("undersampler", undersampler)]

        if undersampler.parameter_grid() is not None:
            for (parameter, values) in undersampler.parameter_grid().items():
                param_grid["undersampler__" + str(parameter)] = values
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
    
    clf = classifiers.fetch_clf(opt.clf)
    if opt.simplified:
        name += "_simplified"
        clf.simplify_model()
    clf.set_params(**{"categorical_features":len(cat_feats)})

    if opt.loss is not None:
        loss = losses.fetch_loss(opt.loss)
        name += f"_{opt.loss}"
        if name.startswith("Base"):
            name = name[5:]
        clf.set_params(**{"loss_fn": loss})
        clf.untoggle_param_grid("loss")
        clf.untoggle_param_grid("clf")

    if opt.ensemble is not None:
        ens = ensembles.fetch_ensemble(opt.ensemble)
        name += f"_{opt.ensemble}"
        ens.set_params(**{"base_estimator": clf})
        if name.startswith("Base"):
            name = name[5:]
        steps = steps + [("ens", ens)]
        for (parameter, values) in ens.parameter_grid().items():
            param_grid["ens__" + str(parameter)] = values
    else:
        if name.startswith("Base"):
            clf.untoggle_param_grid("clf")
        steps = steps + [("clf", clf)]
        for (parameter, values) in clf.parameter_grid().items():
            param_grid["clf__" + str(parameter)] = values
    
    clf.untoggle_param_grid("clf")
    
    pipeline = Pipeline(steps=steps)

    start_time = time.time()
    if c_time is not None:
        start_time -= c_time
    for config in configs:
        pipeline.set_params(**config)
    
    # To do hyperparameter search step by step we must first have a defined classifier
    if name != "Base" and os.path.exists(f"config/{opt.dataset}/Base.pkl") and os.path.exists(f"info/{opt.dataset}/Base.pkl"):
        print(f"Fetching base model best configuration...")
        with open(f"config/{opt.dataset}/Base.pkl", "rb") as fp:
            base_config = pickle.load(fp)
        
        with open(f"info/{opt.dataset}/Base.pkl", "rb") as fp:
            base_info = pickle.load(fp)
            print(f"Adding {base_info['time']} to training time")
            start_time -= base_info["time"]
        clf.set_params(**{key[len("clf__"):]: value for (key, value) in base_config.items()})
        print(f"After fetching base classifier it had parameters: {clf.get_params()}")
    elif name != "Base":
        print(f"Base model best configuration not found. Train base model first.")
        sys.exit()

    
    if "undersampler" in pipeline.named_steps:
        pipeline.named_steps["undersampler"].set_params(**{"estimator": clf})

    search = CustomCV(estimator = pipeline, 
                      param_distributions = param_grid, 
                      n_iter = opt.n_iter,
                      )
                      
    est = search.fit(X_train, y_train)
    end_time = time.time()

    metrics = compute_metrics(est, X_train, y_train, X_test, y_test)

    total_time = end_time - start_time
    
    if opt.loss is None and opt.ensemble is None:
        metric_file = "preprocessing"
    
        subtype = "Base"
        if opt.oversampling and not opt.undersampling:
            subtype = "Oversamplers"
        
        if not opt.oversampling and opt.undersampling:
            subtype = "Undersamplers"

        if opt.oversampling and opt.undersampling:
            subtype = "Hybrid"

    else:
        metric_file = "inprocessing"

        if opt.loss is not None:
            subtype = "Losses"
        
        if opt.ensemble is not None:
            subtype = "Ensembles"

    utils.dump_metrics(metrics, name, metric_file, subtype, search.best_params_, opt.dataset)

    probs = est.predict_proba(X_test)[:, 1]
    fpr, tpr, ths = sklearn.metrics.roc_curve(y_test, probs)
    lower_fpr_idx = np.argmax(fpr[fpr < 0.05])
    th = ths[lower_fpr_idx]

    positives = probs[y_test == 1]
    negatives = probs[y_test == 0]
    plt.hist(probs[y_test == 1], bins = 100, range = (0, 1), label = "Positive", alpha = 0.5, weights = np.ones_like(positives)/float(len(positives)))
    plt.hist(probs[y_test == 0], bins = 100, range = (0, 1), label = "Negative", alpha = 0.5, weights = np.ones_like(negatives)/float(len(negatives)))
    plt.axvline(x = th, alpha = 0.5, color = "red", linestyle = "--", linewidth = 1)
    plt.legend(loc='upper right')
    results_dir = f'results/{opt.dataset}/{metric_file}/{name}/'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'{name}_probs.pdf'))

if __name__ == '__main__':
    main()