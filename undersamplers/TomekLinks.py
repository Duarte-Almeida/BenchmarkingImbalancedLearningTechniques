import numpy as np
from imblearn.under_sampling import TomekLinks
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import _num_features
import FaissKNN

class TomekLinksWrapper(TomekLinks):

    def __init__(self, categorical_features=None, cls = "majority", random_state=42):
        super().__init__(sampling_strategy = cls)
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.kwargs = {}
        self.cls = cls

    def _fit_resample(self, X, y):
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

        # FaissKNN implementation
        nn = FaissKNN.FaissKNN(n_neighbors = 2)
        nn.fit(X_transformed)
        nns = nn.kneighbors(X_transformed, return_distance=False)[:, 1]

        links = self.is_tomek(y, nns, self.sampling_strategy_)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

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
        return {
        }