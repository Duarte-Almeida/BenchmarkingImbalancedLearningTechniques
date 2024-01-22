import numpy as np
from scipy import optimize
from scipy import special
import sklearn.metrics
from skopt import space
import matplotlib.pyplot as plt
from scipy.stats import uniform


class CrossEntropyLoss:

    def __init__(self):
        self.ls_alpha = 0
        self.lr_alpha = 0
        self.weight = 1
        self.grid = {}
        self.grid["ls"] = {"ls_alpha": ("suggest_loguniform", 0.00001, 0.1)}
        self.grid["weighted"] = {"weight":  ("suggest_uniform", 1.0, 1000)}
        self.grid["lr"] = {"lr_alpha": ("suggest_uniform", 0.0, 0.8)}
        self.freeze = [key for key in self.grid.keys()]
        self.kwargs = {}

    def __call__(self, y_true, y_pred):
        if np.isscalar(y_pred):
            y_pred = np.ones_like(y_true) * y_pred
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth_targets(y_true)
        p_wrong = np.where(y_true, 1 - y_pred, y_pred)
        mask = p_wrong <= self.lr_alpha
        loss = -(self.weight * smoothed_y * np.log(y_pred) + (1 - smoothed_y) * np.log(1 - y_pred))
        loss[mask] = 0
        return loss

    def smooth_targets(self, y):
        mean = 0.5
        return (1 - self.ls_alpha) * y + self.ls_alpha * mean

    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth_targets(y_true)
        gradient = y_pred - smoothed_y
        gradient[y_true == 1] *= self.weight
        p_wrong = np.where(y_true, 1 - y_pred, y_pred)
        mask = p_wrong <= self.lr_alpha
        res = gradient.reshape(y_true.shape)
        res[mask] = 0
        return res

    def hess(self, y_true, y_pred):
        hessian = 1e-10 * np.ones_like(y_true)
        return hessian.reshape(y_true.shape)

    def init_score(self, y_true):
        p = np.clip(np.mean(y_true), 1e-15,  1 - 1e-15)
        log_odds = np.log(p / (1 - p))
        return log_odds

    def obj(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        grad = self.grad(y, p)
        return (
         self.grad(y, p), self.hess(y, p))

    def eval(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        is_higher_better = False

        return ('cross_entropy_loss', self(y, p).mean(), is_higher_better)
    
    def adapt_hyperparameters(self, X, y):
        IR = y[y == 0].shape[0] / y[y == 1].shape[0]
        self.grid["weighted"]["weight"] = ("suggest_uniform", 1.0, max(float(np.sqrt(IR)) - 1.0, 1 + 1e-5))

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
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
                res.update(self.grid[name])
        return res

