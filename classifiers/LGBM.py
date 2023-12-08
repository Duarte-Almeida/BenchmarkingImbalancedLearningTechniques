import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
import scipy.special, lightgbm as lgb
import matplotlib.pyplot as plt

class LGBM(BaseEstimator, ClassifierMixin):

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.kwargs = {}

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_fit, self.X_val, self.y_fit, self.y_val = train_test_split(X, y, random_state=42)
        self.model = lgb.LGBMClassifier()
        self.model.set_params(
            objective=self.loss_fn.obj,
            metric='None', 
            learning_rate=0.3, 
            num_iterations=1000, 
            early_stopping_round=20
        )
        self.model.fit(
            self.X_fit,
            self.y_fit,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.log_evaluation(1)],
            eval_metric=(self.loss_fn.eval),
            init_score=np.full_like((self.y_fit), (self.loss_fn.init_score(self.y_fit)), dtype=float),
            eval_init_score=[np.full_like((self.y_val), (self.loss_fn.init_score(self.y_val)), dtype=float)])
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
        scores = scipy.special.expit(self.model.predict(X, raw_score=True) + self.loss_fn.init_score(self.y_fit))
        return np.where(scores <= threshold, 0, 1)

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        probs = scipy.special.expit(self.model.predict(X, raw_score=True) + self.loss_fn.init_score(self.y_fit))
        #print(f"Probs: {probs}")
        #return probs.reshape(1, -1)
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
        grid = {}
        if self.loss_fn.parameter_grid() is not None:
            for key, value in self.loss_fn.parameter_grid().items():
                grid['loss_fn__' + key] = value

        return grid
