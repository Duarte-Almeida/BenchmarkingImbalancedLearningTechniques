from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import copy
import classifiers
import undersamplers
from sklearn.metrics import roc_auc_score
import pickle as pkl
import os
import sys
import scipy.special

class StackedEnsemble:
    def __init__(self, undersampler = undersamplers.RUSWrapper(), sampling_strategy = 1, N = 4, 
                base_estimator = classifiers.LGBM(), meta_learner = classifiers.NeuralNetwork()):
        self.undersampler = undersampler
        self.sampling_strategy = sampling_strategy
        self.undersampler.sampling_ratio = sampling_strategy
        self.N = N
        self.base_estimator = base_estimator
        self.meta_learner = meta_learner
        self.base_estimators = None
        self.stratified_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.kwargs = {}
        self.meta_learner.untoggle_param_grid("all")

    
    def fit(self, X, y):
        self.meta_X = np.zeros((X.shape[0], self.N))
        self.base_estimators = [copy.copy(self.base_estimator) for _ in range(self.N)]
        
        for j in range(self.N):
            #print(f"Subset {j}")
            
            # create j-th subset
            self.undersampler.fit_resample(X, y)

            # subset and oos indices
            selected_idx = self.undersampler.sample_indices_
            set_selected_indices = set(selected_idx)
            oos_idx = np.array([idx for idx in range(X.shape[0]) if idx not in set_selected_indices])
            
            X_j, y_j = X[selected_idx], y[selected_idx]

            curr_clf = self.base_estimators[j]
            for train_index, val_index in self.stratified_kf.split(X_j, y_j):   
                #print(f"A CV fold")
                X_train, y_train = X_j[train_index], y_j[train_index]
                curr_clf.fit(X_train, y_train)
                curr_idx = selected_idx[val_index]
                self.meta_X[curr_idx, j] = curr_clf.predict_proba(X_j[val_index])[:, 1]
                probs = self.meta_X[curr_idx, j]
                y_val = y_j[val_index]
                print(f"AUC: {roc_auc_score(y_val, probs)}")
            
            # fitting the base classifier to the whole subset for OOD prediction
            curr_clf.fit(X_j, y_j)
            curr_idx = oos_idx
            self.meta_X[curr_idx, j] = curr_clf.predict_proba(X[curr_idx])[:, 1]
        
        print("Training meta sampler...")

        # Fitting the meta learner on the meta features
        aux = np.hstack((self.meta_X, y.reshape(-1, 1)))

        self.meta_learner.fit(self.meta_X, y)

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else: 
            threshold = 0.5
       
        scores = self.predict_proba(X)[:, 1]
        return np.where(scores <= threshold, 0, 1)
    
    def predict_proba(self, X):
        meta_X = np.zeros((X.shape[0], self.N))
        
        for i in range(self.N):
            meta_X[:, i] = self.base_estimators[i].predict_proba(X)[:, 1]
            
        res = self.meta_learner.predict_proba(meta_X)
        return res

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if '__' in key:
                idx = key.find('__')
                attr = key[:idx]
                attr_param = key[idx + 2:]
                if attr_param.find("__") == -1:
                    if hasattr(self, attr):
                       getattr(self, attr).set_params(**{attr_param: value})
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.base_estimator.set_params(**{key:value})

        return self

    def parameter_grid(self):
        grid = {
            "sampling_strategy": ("suggest_uniform", 0.0, 1.0),
            "N": ("suggest_int", 2, 10),
        }

        for (param, value) in self.meta_learner.parameter_grid().items():
            grid["meta_learner__" + param] = value

        for (param, value) in self.undersampler.parameter_grid().items():
            grid["undersampler__" + param] = value

        return grid
        
    def adapt_hyperparameters(self, X, y):
        self.base_estimator.adapt_hyperparameters(X, y)