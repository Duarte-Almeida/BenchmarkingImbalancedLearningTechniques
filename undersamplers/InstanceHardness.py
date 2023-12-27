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

class IRWrapper(InstanceHardnessThreshold):

    def __init__(self, categorical_features=None, estimator=None, random_state=42):
        super().__init__()
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.kwargs = {}
        self.estimator = estimator
        self.cv = 3

    def fit_resample(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        skf = StratifiedKFold(n_splits=self.cv)
        probs = np.zeros_like(y, dtype = float)
        #print(self.estimator)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            self.estimator.fit(X[train_index], y[train_index])
            probs += self.estimator.predict_proba(X)[:, 1]
        probs /= self.cv

        # get the 1 - alpha probability quantile
        num_neg = y[y == 0].shape[0]
        num_pos =  y[y == 1].shape[0]      

        prob = num_pos / (num_neg * self.sampling_strategy)
        quantile = np.quantile(probs[y == 0], prob)
        print(f"Which corresponds to quantile {quantile}")

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
            'sampling_strategy': uniform(0, 1)
        }

        return grid