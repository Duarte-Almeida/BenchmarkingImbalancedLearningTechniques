import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
import scipy.special

class CustomRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, loss_fn, verbose=False):
        self.loss_fn = loss_fn
        self.kwargs = {}
        self.verbose = verbose

    def fit(self, X, y, categorical_features=None):
        X, y = check_X_y(X, y)
        if categorical_features is None:
            categorical_features = []

        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_fit, self.X_val, self.y_fit, self.y_val = train_test_split(X, y, random_state=42)

        self.model = RandomForestClassifier(
            random_state=42,
            n_estimators=10,
            # max_depth=None,
            # criterion='gini',
            class_weight='balanced_subsample'
        )

        self.model.fit(
            self.X_fit,
            self.y_fit
        )

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else:
            threshold = 0.5
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        scores = scipy.special.expit(self.model.predict_proba(X)[:, 1] + self.loss_fn.init_score(self.y_fit))
        return np.where(scores <= threshold, 0, 1)

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        probs = scipy.special.expit(self.model.predict_proba(X)[:, 1] + self.loss_fn.init_score(self.y_fit))
        res = np.vstack((1 - probs, probs)).T
        return res

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
            'n_estimators': [50, 100, 200, 1000],  # Number of trees in the forest
            'max_depth': [None, 5, 10, 20],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split

            # Add other hyperparameters as needed
        }
        grid = {}
        if self.loss_fn.parameter_grid() is not None:
            for key, value in self.loss_fn.parameter_grid().items():
                grid['loss_fn__' + key] = value

        return grid
