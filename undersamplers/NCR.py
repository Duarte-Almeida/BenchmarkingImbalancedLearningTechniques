import numpy as np
import pandas as pd
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import _num_features
import FaissKNN

class NCRWrapper(NeighbourhoodCleaningRule):

    def __init__(self, categorical_features=None, cls = "majority", random_state=42):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.cls = cls
        self.kwargs = {}
        super().__init__(n_neighbors = FaissKNN.FaissKNN(), kind_sel = "mode", sampling_strategy = cls)

    def fit_resample(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self.n_features_ = _num_features(X)
        random_state = check_random_state(self.random_state)

        if self.categorical_features:

            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_cat = self.encoder.fit_transform(X[:, -self.categorical_features:])
            X_non_cat = X[:, :-self.categorical_features]

            std_devs = np.std(X_non_cat, axis = 1)
            X_cat = X_cat * (np.median(std_devs) / np.sqrt(2))
            
            X_transformed = np.concatenate((X_non_cat, X_cat), axis=1)
        else:
            X_transformed = X

        X_resampled = super().fit_resample(X_transformed, y)
        X_resampled = X[self.sample_indices_]
        y_resampled = y[self.sample_indices_]
        
        return X_resampled, y_resampled

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
            else:
                self.kwargs[key] = value

        for attr, attr_dict in attr_params.items():
            getattr(self, attr).set_params(**attr_dict)

        return self


    def parameter_grid(self):
        grid = {
        }
        if self.n_neighbors.parameter_grid() is not None:
            for key, value in self.n_neighbors.parameter_grid().items():
                grid['n_neighbors__' + key] = value

        return grid
    
    def adapt_hyperparameters(self, X, y):
        pass