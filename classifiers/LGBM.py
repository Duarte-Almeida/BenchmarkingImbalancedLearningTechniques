import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
import scipy.special, lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import loguniform, uniform, randint
import time
from skopt.space import Real, Integer
import losses
import sys
from functools import *
from scipy import special
import matplotlib.pyplot as plt

attribute_names = {
    "loss_fn":"objective"
}

class LGBM(BaseEstimator, ClassifierMixin):

    def __init__(self, loss_fn = losses.CrossEntropyLoss(), verbose = False, simplified = False, **kwargs):
        self.loss_fn = loss_fn
        self.kwargs = {}
        self.verbose = verbose
        self.simplified = simplified
        if self.verbose and self.verbose > 0:
            self.callbacks = [lgb.log_evaluation(1)]
        else:
            self.callbacks = []

        self.model = lgb.LGBMClassifier()
        self.model.set_params(
            metric = None,
            objective=self.loss_fn.obj,
            learning_rate=0.3, 
            n_estimators=1000, 
            early_stopping_round=20,
            verbose = -1,
            min_child_weight = 1e-20,
        )

        self.grid = {}  
        
        self.grid["clf"] = {
            'model__learning_rate': ("suggest_loguniform", 0.01, 0.50),  # Learning rate
            'model__num_leaves': ("suggest_int", 10, 201),          # Maximum number of leaves in one tree
            'model__max_depth': ("suggest_int", -1, 21),  # Maximum depth of the tree
            'model__min_child_samples': ("suggest_int", 20, 101),  # Minimum number of data needed in a child
            'model__subsample': ("suggest_uniform", 0.5, 1),  # Subsample ratio of the training instance
            'model__colsample_bytree': ("suggest_uniform", 0.5, 1),  # Subsample ratio of columns when constructing each tree
            'model__n_estimators': ("suggest_int", 50, 1000),  # Number of boosting rounds
            'model__reg_lambda': ("suggest_loguniform", 10.0, 10000.0)  # Regularization lambda
        }

        
        self.grid["loss"] = {}
        if self.loss_fn.parameter_grid() is not None:
            for key, value in self.loss_fn.parameter_grid().items():
                self.grid["loss"]['model__loss_fn__' + key] = value

        self.freeze = [key for key in self.grid.keys()]
        self.set_params(**kwargs)

    def simplify_model(self):
        self.grid["clf"]['model__n_estimators'] = ("suggest_int", 50, 100)
        self.simplified = True
    
    def adapt_hyperparameters(self, X, y, categorical_features = None):

        print(f"Adapting hyperparameters")
        self.loss_fn.adapt_hyperparameters(X, y)

        return 

    def fit(self, X, y, categorical_features = None):

        X, y = check_X_y(X, y)
        if categorical_features is None:
            categorical_features = 0
            
        self.model.set_params(objective = self.loss_fn.obj)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_fit, self.X_val, self.y_fit, self.y_val = train_test_split(X, y, random_state=42)
        self.model.fit(
            self.X_fit,
            self.y_fit,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=self.callbacks,
            eval_metric=(self.loss_fn.eval),
            init_score=np.full_like((self.y_fit), (self.loss_fn.init_score(self.y_fit)), dtype=float),
            eval_init_score=[np.full_like((self.y_val), (self.loss_fn.init_score(self.y_val)), dtype=float)],
            categorical_feature = [i for i in range(X.shape[1] - categorical_features, X.shape[1])]),
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
        start = time.time()
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        probs = scipy.special.expit(self.model.predict(X, raw_score=True) + self.loss_fn.init_score(self.y_fit))
        res = np.vstack((1 - probs, probs)).T
        end = time.time()

        return res

    def get_params(self, deep=True):
        res = {}
        res.update({"model__" + key: value for (key, value) in self.model.get_params().items()})

        res.update({param: getattr(self, param)
                for param in self._get_param_names()})
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
                    getattr(self, attr).set_params(**{attr_param:value})
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self
    
    def toggle_param_grid(self, name):
        if name == "all":
            self.freeze = [key for key in self.grid.keys()]
        elif name in self.grid:
            if name not in self.freeze:
                self.freeze.append(name)
    
    def untoggle_param_grid(self, name):
        if name == "all":
            self.freeze = []
        if name in self.grid.keys() and name in self.freeze:
            self.freeze.remove(name)
        
    def parameter_grid(self):
        res = {}
        for name in self.grid.keys():
            if name not in self.freeze:
                # Generate parameter grid dynamically
                if name == "loss":
                    res.update({"loss_fn__" + key: value for (key, value) in self.loss_fn.parameter_grid().items()})
                else:
                    res.update(self.grid[name])
        return res
