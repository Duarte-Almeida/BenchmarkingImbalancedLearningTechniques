import numpy as np
import pandas as pd
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import _num_features
import FaissKNN
from scipy.stats import uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
import copy
import matplotlib.pyplot as plt
import sys
from skopt import space

class IHTWrapper(InstanceHardnessThreshold):

    def __init__(self, categorical_features=None, cls = "majority", estimator=None, random_state=42):
        super().__init__()
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.kwargs = {}
        self.estimator = estimator
        self.cv = 3
        self.cls = cls
        self.sampling_ratio = 1.0

    def fit_resample(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        skf = StratifiedKFold(n_splits=self.cv)
        probs = np.zeros_like(y, dtype = float)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            self.estimator.fit(X[train_index], y[train_index])
            probs += self.estimator.predict_proba(X)[:, 1]
        probs /= self.cv

        neg = y[y == 0].shape[0]
        pos = y[y == 1].shape[0]
        IR = pos / neg
        eps = 1 / pos

        rng = np.random.default_rng(self.random_state)   

        if self.cls == "majority":
            self.sampling_strategy = min(1, IR + self.sampling_ratio * (1 - IR) + eps)
            num_samples = int(pos / self.sampling_ratio)
            prob = num_samples / neg
            quantile = np.quantile(probs[y == 0], prob)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where((y == 0) & (probs <= quantile))[0]
            idx = np.concatenate((neg_idx, pos_idx))
            self.sample_indices_ = idx
            return X[idx], y[idx]
            
        else:
            self.sampling_strategy = min(1, (1 - self.sampling_ratio) + eps)
            num_samples_pos = int(pos * self.sampling_strategy)
            num_samples_neg = int(neg * self.sampling_strategy)

            neg_idx = np.where(y == 0)[0]
            pos_idx = np.where(y == 1)[0]

            pos_prob = num_samples_pos / pos
            pos_quantile = np.quantile(probs[y == 1], 1 - pos_prob)

            neg_prob = num_samples_neg / neg
            neg_quantile = np.quantile(probs[y == 0], neg_prob)

            pos_idx = np.where((y == 1) & (probs >= pos_quantile))[0]
            neg_idx = np.where((y == 0) & (probs <= neg_quantile))[0]
            idx = np.concatenate((neg_idx, pos_idx))
            self.sample_indices_ = idx
            return X[idx], y[idx]

        # remove most difficult majority instances
        idx = np.where((y == 1) | ((y == 0) & (probs <= quantile)))[0]
        res = y[idx]
        return X[idx], y[idx]


    def set_params(self, **params):
        if not params:
            return self
        attr_params = {}
        for key, value in params.items():
            if '__' in key:
                idx = key.find('__')
                attr = key[:idx]
                attr_param = key[idx + 2:]
                if hasattr(self, attr):
                    if attr not in attr_params:
                        attr_params[attr] = {}
                    attr_params[attr][attr_param] = value
            if hasattr(self, key):
                setattr(self, key, value)
                if key == "estimator":
                    self.estimator = copy.deepcopy(value)
            else:
                self.kwargs[key] = value

        for attr, attr_dict in attr_params.items():
            getattr(self, attr).set_params(**attr_dict)

        return self


    def parameter_grid(self):
        grid = {
            'sampling_ratio': ("suggest_uniform", 0.0, 0.1),
        }

        return grid
    
    def adapt_hyperparameters(self, X, y):
        pass