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

    def __init__(self, loss_fn = losses.CrossEntropyLoss(), verbose = False):
        self.loss_fn = loss_fn
        self.kwargs = {}
        self.verbose = verbose
        if self.verbose:
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
            'model__num_leaves': ("suggest_int", 10, 201),  # Maximum number of leaves in one tree
            'model__max_depth': ("suggest_int", -1, 21),  # Maximum depth of the tree
            'model__min_child_samples': ("suggest_int", 20, 101),  # Minimum number of data needed in a child
            'model__subsample': ("suggest_uniform", 0.5, 1),  # Subsample ratio of the training instance
            'model__colsample_bytree': ("suggest_uniform", 0.5, 1),  # Subsample ratio of columns when constructing each tree
            'model__n_estimators': ("suggest_int", 500, 5000),  # Number of boosting rounds
            'model__reg_lambda': ("suggest_loguniform", 10.0, 10000.0)  # Regularization lambda
        }
        
        self.grid["loss"] = {}
        if self.loss_fn.parameter_grid() is not None:
            for key, value in self.loss_fn.parameter_grid().items():
                self.grid["loss"]['model__loss_fn__' + key] = value
        #print(self.grid)
        self.freeze = [key for key in self.grid.keys()]
    
    def adapt_hyperparameters(self, X, y, categorical_features = None):

        print(f"Adapting hyperparameters")

        self.loss_fn.adapt_hyperparameters(X, y)

        return 

        X, y = check_X_y(X, y)
        if categorical_features is None:
            categorical_features = 0

        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_fit, self.X_val, self.y_fit, self.y_val = train_test_split(X, y, random_state=42)

        predictions_list = [[], []]

        def get_predictions(cl_model, pred_list):
            it = cl_model.iteration
            predictions = cl_model.model.predict(self.X_fit, raw_score=True)
            pred_list[it].append(predictions.copy())

        f = partial(get_predictions, pred_list = predictions_list)

        lambda_values = np.arange(1, 1000, 100).astype(int)
        old_num_estimators = self.model.n_estimators
        self.model.set_params(n_estimators = 2)
        self.model.set_params(verbose = -1)
        self.model.set_params(objective = self.loss_fn.obj)

        for l in lambda_values:
            self.model.set_params(reg_lambda = l)

            self.model.fit(
                self.X_fit,
                self.y_fit,
                verbose_eval = -1,
                eval_set=[(self.X_val, self.y_val)],
                callbacks= [f],
                eval_metric=(self.loss_fn.eval),
                init_score=np.full_like((self.y_fit), (self.loss_fn.init_score(self.y_fit)), dtype=float),
                eval_init_score=[np.full_like((self.y_val), (self.loss_fn.init_score(self.y_val)), dtype=float)],
                categorical_feature = [i for i in range(X.shape[1] - categorical_features, X.shape[1])]),
        
        predictions_list = np.array(predictions_list)
        
        first_predictions = special.expit(predictions_list[0])
        final_predictions = special.expit(predictions_list[1])
        second_predictions = predictions_list[1] - predictions_list[0]
    
        actual_loss = np.apply_along_axis(lambda row: self.loss_fn(self.y_fit, row), axis = 1, arr = final_predictions)
        init_loss = np.apply_along_axis(lambda row: self.loss_fn(self.y_fit, row), axis = 1, arr = first_predictions)
        gradients = np.apply_along_axis(lambda row: self.loss_fn.grad(self.y_fit, row), axis = 1, arr = first_predictions)
        
        errors = np.mean((actual_loss - (init_loss + gradients * second_predictions)) ** 2, axis = 1)
        second_derivative = errors[2:] + errors[:-2] - 2 * errors[1:-1]
        gains = np.mean(init_loss - actual_loss, axis = 1)
        elbow_idx = np.argwhere(second_derivative == np.max(second_derivative))[0][0] + 1

        best_lambda = lambda_values[np.argwhere(gains == np.max(gains[elbow_idx:]))[0]][0]
        print(f"Best value found for lambda: {best_lambda}")
        self.grid["model__reg_lambda"] = uniform(float(best_lambda - 50), 100)
        self.model.set_params(n_estimators = old_num_estimators)
        

    def fit(self, X, y, categorical_features = None):

        X, y = check_X_y(X, y)
        if categorical_features is None:
            categorical_features = 0

        print(f"Using the following parameters: \n {self.model.get_params()}")
            
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

    def set_params(self, **params):
        if not params:
            return self
        attr_params = {}
        for key, value in params.items():
            #print(f"Will set {key} to {value}")
            if '__' in key:
                idx = key.find('__')
                attr = key[:idx]
                attr_param = key[idx + 2:]
                if attr_param.find("__") == -1:
                    if hasattr(self, attr):
                        if attr not in attr_params:
                            attr_params[attr] = {}
                        attr_params[attr][attr_param] = value
                else:
                    aux_idx = attr_param.find("__")
                    second_attr = attr_param[:aux_idx]
                    attr_param = attr_param[aux_idx + 2:]
                    print(f"Well, I am here to set {second_attr} to {attr_param}")
                    if hasattr(self, second_attr) and hasattr(self, attr):
                        getattr(self, second_attr).set_params(**{attr_param: value})
                        #getattr(self, attr).set_params(**{attribute_names[attr_param]: value})
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        for attr, attr_dict in attr_params.items():
            print(f"Setting {attr_dict} to {attr}")
            getattr(self, attr).set_params(**attr_dict)

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
            #print(f"Done!")
            self.freeze.remove(name)
            #print(f"Here is my frozen stuff: {self.freeze}")
        
    def parameter_grid(self):
        res = {}
        for name in self.grid.keys():
            if name not in self.freeze:
                if name == "loss":
                    res.update({"loss_fn__" + key: value for (key, value) in self.loss_fn.parameter_grid().items()})
                else:
                    res.update(self.grid[name])
        #print(f"Here is the result: {res}")
        return res
