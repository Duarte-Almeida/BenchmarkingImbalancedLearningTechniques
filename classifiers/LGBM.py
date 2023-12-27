import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
import scipy.special, lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import loguniform, uniform, randint
import time

class LGBM(BaseEstimator, ClassifierMixin):

    def __init__(self, loss_fn, verbose = False):
        self.loss_fn = loss_fn
        self.kwargs = {}
        self.verbose = verbose
        if self.verbose:
            self.callbacks = [lgb.log_evaluation(1)]
        else:
            self.callbacks = []

    def fit(self, X, y, categorical_features = None):
        X, y = check_X_y(X, y)
        if categorical_features is None:
            categorical_features = 0
        #else:
        #    print(f"index of categorical features: {categorical_features}")
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_fit, self.X_val, self.y_fit, self.y_val = train_test_split(X, y, random_state=42)
        self.model = lgb.LGBMClassifier()
        self.model.set_params(
            objective=self.loss_fn.obj,
            metric='None', 
            learning_rate=0.3, 
            n_estimators=1000, 
            early_stopping_round=20,
            verbose = -1,
        )
        self.model.fit(
            self.X_fit,
            self.y_fit,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=self.callbacks,
            eval_metric=(self.loss_fn.eval),
            init_score=np.full_like((self.y_fit), (self.loss_fn.init_score(self.y_fit)), dtype=float),
            eval_init_score=[np.full_like((self.y_val), (self.loss_fn.init_score(self.y_val)), dtype=float)],
            categorical_feature = [i for i in range(X.shape[1] - categorical_features, X.shape[1])])
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else: # default threshold of 0.4
            threshold = 0.5
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        scores = scipy.special.expit(self.model.predict(X, raw_score=True) + self.loss_fn.init_score(self.y_fit))
        return np.where(scores <= threshold, 0, 1)

    def predict_proba(self, X):
        #print(f"Predicting proba...")
        start = time.time()
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        probs = scipy.special.expit(self.model.predict(X, raw_score=True) + self.loss_fn.init_score(self.y_fit))
        res = np.vstack((1 - probs, probs)).T
        end = time.time()
        #print(f"Predicted in {end-start}")
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
            'learning_rate': loguniform(0.01, 0.5),  # Learning rate
            'num_leaves': randint(10, 200),  # Maximum number of leaves in one tree
            'max_depth': randint(-1, 10),  # Maximum depth of the tree
            'min_child_samples': randint(20, 100),  # Minimum number of data needed in a child
            'subsample': uniform(0.5, 1.0),  # Subsample ratio of the training instance
            'colsample_bytree': uniform(0.5, 1.0),  # Subsample ratio of columns when constructing each tree
            'reg_alpha': uniform(0, 2),  # L1 regularization term on weights
            'reg_lambda': uniform(0, 2),  # L2 regularization term on weights
            'n_estimators': uniform(50, 500),  # Number of boosting rounds
        }
        #grid = {}
        if self.loss_fn.parameter_grid() is not None:
            for key, value in self.loss_fn.parameter_grid().items():
                grid['loss_fn__' + key] = value

        return grid
