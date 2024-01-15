import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import _num_features
import FaissKNN
from imblearn.utils import check_sampling_strategy
import pandas as pd
from scipy.stats import uniform
from skopt import space

class SMOTEWrapper(SMOTE):
    def __init__(self, categorical_features=0, random_state=42):
        super().__init__(k_neighbors = FaissKNN.FaissKNN())
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.kwargs = {}
        self.sampling_ratio = 1

    def fit_resample(self, X, y):

        neg = y[y == 0].shape[0]
        pos = y[y == 1].shape[0]
        IR = pos / neg
        eps = 1 / pos

        #print(f"NN: {self.k_neighbors} \n sampling_ratio: {self.sampling_ratio}")
        self.sampling_strategy = min(1, IR + self.sampling_ratio * (1 - IR) + eps)
        #print(f"NN: {self.k_neighbors} \n sampling_strategy: {self.sampling_strategy}")

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        self.n_features_ = _num_features(X)
        random_state = check_random_state(self.random_state)

        if self.categorical_features > 0:

            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_cat = self.encoder.fit_transform(X[:, -self.categorical_features:])
            X_non_cat = X[:, :-self.categorical_features]

            std_devs = np.std(X_non_cat, axis = 1)
            X_cat = X_cat * (np.median(std_devs) / np.sqrt(2))
            

            X_transformed = np.concatenate((X_non_cat, X_cat), axis=1)
        else:
            X_transformed = X.astype(float)

        X_resampled, y_resampled = super().fit_resample(X_transformed, y)
       

        if self.categorical_features:
            X_cat_resampled = X_resampled[:, -X_cat.shape[1]:]
            X_non_cat_resampled = X_resampled[:, :-X_cat.shape[1]]

            X_cat_resampled_inv = self.encoder.inverse_transform(X_cat_resampled)

            X_resampled = np.concatenate((X_non_cat_resampled, X_cat_resampled_inv), axis=1)
 
        return X_resampled, y_resampled

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y=None):

        rng = check_random_state(self.random_state)
        X_new = super()._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        #print(X_new)
    
        if self.categorical_features > 0:
            all_neighbors = nn_data[nn_num[rows]]
            categories_size = [self.n_features_ - self.categorical_features] + [
                cat.size for cat in self.encoder.categories_
            ]

            for start_idx, end_idx in zip(
                np.cumsum(categories_size)[:-1], np.cumsum(categories_size)[1:]
            ):
                col_maxs = all_neighbors[:, :, start_idx:end_idx].sum(axis=1)
                # tie breaking argmax
                is_max = np.isclose(col_maxs, col_maxs.max(axis=1, keepdims=True))
                max_idxs = rng.permutation(np.argwhere(is_max))
                xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
                col_sels = max_idxs[idx_sels, 1]

                ys = start_idx + col_sels
                X_new[:, start_idx:end_idx] = 0
                X_new[xs, ys] = 1

        return X_new

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._get_param_names()}
    
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
            'sampling_ratio': ("suggest_uniform", 0.0, 0.1)
        }
        if self.k_neighbors.parameter_grid() is not None:
            for key, value in self.k_neighbors.parameter_grid().items():
                grid['k_neighbors__' + key] = value

        return grid

    def adapt_hyperparameters(self, X, y):
        pass

