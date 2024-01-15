import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import _num_features
from sklearn.utils import _safe_indexing
import FaissKNN
from scipy import sparse
from imblearn.utils import check_sampling_strategy
from scipy.stats import uniform
from skopt import space

class ADASYNWrapper(ADASYN):

    def __init__(self, categorical_features = 0, random_state=42):
        super().__init__(n_neighbors = FaissKNN.FaissKNN())
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.kwargs = {}
        self.sampling_ratio =1

    def fit_resample(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        neg = y[y == 0].shape[0]
        pos = y[y == 1].shape[0]
        IR = pos / neg
        eps = 1 / pos

        self.sampling_strategy = min(1, IR + self.sampling_ratio * (1 - IR) + eps)
        self.sampling_strategy_ = check_sampling_strategy(self.sampling_strategy, y, "over-sampling")
        
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
            X_transformed = X

        X_resampled, y_resampled = self.fit_resample_aux(X_transformed, y)

        if self.categorical_features > 0:
            X_cat_resampled = X_resampled[:, -X_cat.shape[1]:]
            X_non_cat_resampled = X_resampled[:, :-X_cat.shape[1]]

            X_cat_resampled_inv = self.encoder.inverse_transform(X_cat_resampled)

            X_resampled = np.concatenate((X_non_cat_resampled, X_cat_resampled_inv), axis=1)
        
        return X_resampled, y_resampled
    
    def fit_resample_aux(self, X, y):

        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)
            num_cat_feats = len(self.encoder.get_feature_names_out())
            X_non_cat = X[:, :-num_cat_feats]
            X_cat = X[:, -num_cat_feats:]

            self.nn_.fit(X)

            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            #print(y)
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            n_neighbors = self.nn_.n_neighbors - 1
            ratio_nn = np.sum(y[nns] != class_sample, axis=1) / n_neighbors
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neigbours belong to the majority"
                    " class. This case will induce a NaN case"
                    " with a division by zero. ADASYN is not"
                    " suited for this specific dataset."
                    " Use SMOTE instead."
                )
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            # rounding may cause new amount for n_samples
            n_samples = np.sum(n_samples_generate)
            if not n_samples:
                raise ValueError(
                    "No samples will be generated with the provided ratio settings."
                )

            # the nearest neighbors need to be fitted only on the current class
            # to find the class NN to generate new samples
            self.nn_.fit(X_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]

            enumerated_class_indices = np.arange(len(target_class_indices))
            rows = np.repeat(enumerated_class_indices, n_samples_generate)
            cols = random_state.choice(n_neighbors, size=n_samples)
            diffs = X_non_cat[nns[rows, cols]] - X_non_cat[rows]
            steps = random_state.uniform(size=(n_samples, 1))

            # generate continuous part of new instances
            if sparse.issparse(X):
                sparse_func = type(X).__name__
                steps = getattr(sparse, sparse_func)(steps)
                X_non_cat_new = X_non_cat[rows] + steps.multiply(diffs)
            else:
                X_non_cat_new = X_non_cat[rows] + steps * diffs
            
            rng = check_random_state(self.random_state)

            categories_size = [0] + [
                cat.size for cat in self.encoder.categories_
            ]

            X_cat_new = X_cat[rows].copy()
            all_neighbors = X_cat[nns[rows]]
            for start_idx, end_idx in zip(
                np.cumsum(categories_size)[:-1], np.cumsum(categories_size)[1:]
            ):  
                #print(f"From {start_idx} to {end_idx}")
                col_maxs = all_neighbors[:, :, start_idx:end_idx].sum(axis=1)
                # tie breaking argmax
                is_max = np.isclose(col_maxs, col_maxs.max(axis=1, keepdims=True))
                max_idxs = rng.permutation(np.argwhere(is_max))
                xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
                col_sels = max_idxs[idx_sels, 1]

                ys = start_idx + col_sels
                X_cat_new[:, start_idx:end_idx] = 0
                X_cat_new[xs, ys] = 1

            if sparse.issparse(X):
                X_new = sparse.hstack((X_non_cat_new, X_cat_new), format=X.format)
            else:
                X_new = np.hstack((X_non_cat_new, X_cat_new))
            
            X_new = X_new.astype(X.dtype)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            y_resampled.append(y_new)


        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled


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
        if self.n_neighbors.parameter_grid() is not None:
            for key, value in self.n_neighbors.parameter_grid().items():
                grid['n_neighbors__' + key] = value

        return grid
    
    def adapt_hyperparameters(self, X, y):
        pass



