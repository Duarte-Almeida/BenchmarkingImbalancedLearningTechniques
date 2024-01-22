import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, PredefinedSplit, RandomizedSearchCV
from itertools import product
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
import optuna
import matplotlib.pyplot as plt
from functools import partial
import thresholds
import pickle as pkl

import sys


class CustomCV():
    '''
    Custom Hyperparameter search with TPE
    '''

    def __init__(self, estimator, param_distributions, n_iter,  *args, **kwargs):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = None
        self.n_iter = n_iter
       
    def fit(self, X, y, *args, **kwargs):

        # For safety, normalize both X and y to numpy arrays if they are pandas datafarmes
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # hyperparameter search based on ROC-AUC
        def objective(trial, X_aux, y_aux, param_grid, cv_object, estimator):
            params = {param: getattr(trial, value[0])(param, *value[1:]) for (param, value) in param_grid.items()}
            scores = []

            for i, (train_index, test_index) in enumerate(cv_object):

                X_train, y_train = X_aux[train_index], y_aux[train_index]
                X_test, y_test = X_aux[test_index], y_aux[test_index]
               
                estimator.set_params(**params)
                # try:
                estimator.fit(X_train, y_train, *args, **kwargs)
                # except Exception:   # assign score of 0 in case of crash (but not error)
                #     return 0

                probs = estimator.predict_proba(X_test)[:, 1]
                scores.append(roc_auc_score(y_test, probs))
                print(f"Score obtained on {i}-th fold: {scores[-1]}")

            return np.mean(scores)
        
        # perform 5 fold cross validation
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        self.cv = [idx for idx in skf.split(X, y)]
    
        # Some hyperparameters are dependent on the input
        for (name, step) in self.estimator.steps:
            step.adapt_hyperparameters(X, y)
            for (parameter, values) in step.parameter_grid().items():
                self.param_distributions[f"{name}__" + str(parameter)] = values

        clf_objective = partial(objective, X_aux = X, y_aux = y, param_grid = self.param_distributions, 
                                cv_object = self.cv, estimator = self.estimator)
            
        print(f"Finding best hyperparameter combinations...")
        print(self.estimator.get_params())
        study = optuna.create_study(direction='maximize', sampler = optuna.samplers.TPESampler(seed = 42))
        #study.enqueue_trial({key: value for (key, value) in self.estimator.get_params().items() \
        #                    if key in self.param_distributions.keys()})
        study.optimize(clf_objective, n_trials = self.n_iter)

        self.best_score_ = study.best_value
        self.best_params_ = study.best_params

        print("Best parameter for classifier (CV score=%0.3f):" %  self.best_score_)
        print(f"{self.best_params_}")

        # refit the estimator to the best hyperparameters
        self.best_estimator_ = self.estimator
        for (attr, value) in self.best_params_.items():
            idx = attr.find("__")
            component = attr[:idx]
            attr_name = attr[idx + 2:]
            self.best_estimator_.named_steps[component].set_params(**{attr_name: value})

        #self.best_estimator_.fit(X, y)
        return self.best_estimator_