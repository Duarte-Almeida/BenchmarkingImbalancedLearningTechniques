import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, PredefinedSplit, RandomizedSearchCV
from itertools import product
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
import optuna
import matplotlib.pyplot as plt
from functools import partial
import thresholds

import sys

def get_train_test_splits(X, y, oversampler, undersampler):
    #print("Generating folds")
    n_splits = 5
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    X_final = X.copy()
    y_final = y.copy()
    splits = []
    num_resampled = 0
    counter = X.shape[0]

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_aux, y_aux = X[train_index, :], y[train_index]
        orig_shape = X_aux.shape[0]
        #print(f"Original number of samples: {orig_shape}")
        sampled_indices = np.arange(X_aux.shape[0]).astype(int)

        if oversampler is not None:
            X_aux, y_aux = oversampler.fit_resample(X_aux, y_aux)
            sampled_indices = np.arange(X_aux.shape[0]).astype(int)
                
        if undersampler is not None:
            X_aux, y_aux = undersampler.fit_resample(X_aux, y_aux)
            sampled_indices = undersampler.sample_indices_
        #print(f"Original number of samples: {orig_shape}")
        orig_train_indices = np.where(sampled_indices < orig_shape)[0]
        resampled_train_indices = np.where(sampled_indices >= orig_shape)[0]

        X_r, y_r = X_aux[resampled_train_indices], y_aux[resampled_train_indices]
        X_final = np.concatenate((X_final, X_r), axis = 0)
        y_final = np.concatenate((y_final, y_r), axis = 0)

        train_index = train_index[orig_train_indices]
        #print(f"Original train_indices: {train_index}")
        #print(f"New train_indices: {np.arange(counter, counter + X_r.shape[0])}")
        train_index = np.concatenate((train_index, np.arange(counter, counter + X_r.shape[0]))) 
        counter += X_r.shape[0]

        splits.append((train_index, test_index))   
       
    return X_final, y_final, splits


class CustomCV():
    '''
    Custom GridSearchCV wrapper that freezes the preprocessor configuration (e.g., oversampler)
    and then tries out parameter combinations of downstream components to speed up cross validation
    '''

    def __init__(self, clf, estimator, param_distributions, n_iter, prep_param_distributions = {}, prep_n_iter = [], *args, **kwargs):
        self.estimator = estimator
        self.clf = clf
        self.param_distributions = param_distributions
        self.prep_param_distributions = prep_param_distributions
        self.prep_search = []

        self.search = RandomizedSearchCV(estimator = clf, param_distributions = param_distributions,
                                    #optimizer_kwargs = {"base_estimator" : "DUMMY"},
                                    n_iter = n_iter, refit = False, *args, **kwargs)
        if self.prep_param_distributions != {}:
            self.prep_search =RandomizedSearchCV(estimator = estimator, param_distributions = prep_param_distributions, 
                                            #optimizer_kwargs = {"base_estimator" : "DUMMY"},
                                            n_iter = prep_n_iter, refit = False, *args, **kwargs)
      
    def fit(self, X, y, *args, **kwargs):

        oversampler = None
        undersampler = None

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        def objective(trial, X_aux, y_aux, param_grid, cv_object, estimator):
            params = {param: getattr(trial, value[0])(param, *value[1:]) for (param, value) in param_grid.items()}
            scores = []

            #print(f"Trying out {params}")

            for i, (train_index, test_index) in enumerate(cv_object):
                #print("here 1")
                X_train, y_train = X_aux[train_index], y_aux[train_index]
                X_test, y_test = X_aux[test_index], y_aux[test_index]
                #print("here 2")
                estimator.set_params(**params)
                try:
                #print("here 3")
                    estimator.fit(X_train, y_train, *args, **kwargs)
                #print("here 4")
                except Exception:
                    return 0

                probs = estimator.predict_proba(X_test)[:, 1]
                scores.append(roc_auc_score(y_test, probs))
                print(f"Score obtained on {i}-th fold: {scores[-1]}")

            return np.mean(scores)
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        cv = [idx for idx in skf.split(X, y)]
    
        # two step randomized cross validation
        # get best preprocessor parameters given fixed custom classifier
        if self.prep_param_distributions != {}:
            print(f"Finding best preprocessor...")
            #self.prep_search.fit(X, y, *args, **kwargs)
            prep_objective = partial(objective, X_aux = X, y_aux = y, param_grid = self.prep_param_distributions, 
                                 cv_object = cv, estimator = self.estimator)
            prep_study = optuna.create_study(direction='maximize', sampler = optuna.samplers.TPESampler(seed = 42))
            prep_study.optimize(prep_objective, n_trials=self.search.n_iter)

            self.prep_search.best_score_ = prep_study.best_value
            self.prep_search.best_params_ = prep_study.best_params

            if "oversampler" in self.estimator.named_steps.keys():
                oversampler = self.estimator.named_steps["oversampler"]
            if "undersampler" in self.estimator.named_steps.keys():
                undersampler = self.estimator.named_steps["undersampler"]
            for (param, value) in self.prep_search.best_params_.items():
                if param.startswith("oversampler"):
                    attr = param[len("oversampler") + 2:]
                    oversampler.set_params(**{attr: value})
                if param.startswith("undersampler"):
                    attr = param[len("undersampler") + 2:]
                    undersampler.set_params(**{attr: value})
                    
            #self.search.estimator.set_params(**self.prep_search.best_params_)
            print(f"Best preprocessor config: {self.prep_search.best_params_}")
            print(f"with score: {self.prep_search.best_score_}")
        
        else:
            #X_tr, y_tr, cv = get_train_test_splits(X, y, oversampler, undersampler)
            setattr(self.search, "cv", cv)

            self.search.estimator.named_steps["clf"].adapt_hyperparameters(X, y)
            for (parameter, values) in self.search.estimator.named_steps["clf"].parameter_grid().items():
                self.param_distributions["clf__" + str(parameter)] = values
            #print(self.param_distributions)

            clf_objective = partial(objective, X_aux = X, y_aux = y, param_grid = self.param_distributions, 
                                 cv_object = cv, estimator = self.clf)
                
            # get best classifier given previously obtained best estimator
            print(f"Finding best classifier...")
            #search_space = self.search.estimator.named_steps["clf"].define_search_space()
            #self.search.fit(X_tr, y_tr, *args, **kwargs)
            study = optuna.create_study(direction='maximize', sampler = optuna.samplers.TPESampler(seed = 42))
            study.optimize(clf_objective, self.search.n_iter)

            self.search.best_score_ = study.best_value
            self.search.best_params_ = study.best_params

            print("Best parameter for classifier (CV score=%0.3f):" %  self.search.best_score_)
            print(f"{self.search.best_params_}")
            #sys.exit()

        if oversampler is not None or undersampler is not None:
            self.best_estimator_ = self.estimator
            print(self.best_estimator_.get_params())
            for (attr, value) in self.prep_search.best_params_.items():
                idx = attr.find("__")
                component = attr[:idx]
                attr_name = attr[idx + 2:]
                self.best_estimator_.named_steps[component].set_params(**{attr_name: value})
        else:
            self.best_estimator_ = self.search.estimator
            for (attr, value) in self.search.best_params_.items():
                idx = attr.find("__")
                component = attr[:idx]
                attr_name = attr[idx + 2:]
                self.best_estimator_.named_steps[component].set_params(**{attr_name: value})

        #self.best_estimator_.fit(X[:-200], y[:-200])
        #probs = self.best_estimator_.predict_proba(X[:-200])[:, 1]
        #print(roc_auc_score(y[:-200], probs))

        if oversampler is not None or undersampler is not None:
            #self.prep_search.best_params_.update(self.search.best_params_)
            self.best_params_ =self.prep_search.best_params_
        else:
            self.best_params_ = self.search.best_params_

        ths = []
        for i, (train_index, test_index) in enumerate(cv):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            self.best_estimator_.fit(X_train, y_train)
            probs = self.best_estimator_.predict_proba(X_test)[:, 1]

            ths.append(thresholds.compute_FPR_cutoff(y_test, probs))
            print(f"Predicted thresholds {ths[-1]}")
        
        self.th = np.mean(ths)

        self.best_estimator_.fit(X, y)
        print(f"Best overall config: {self.best_params_}")
        return self.best_estimator_