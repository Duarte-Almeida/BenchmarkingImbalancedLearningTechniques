import numpy as np
from scipy import optimize
from scipy import special
import sklearn.metrics
from skopt import space
import matplotlib.pyplot as plt
import sys
from scipy.stats import uniform

class GradientHarmonizedLoss:

    def __init__(self, alpha = 0, beta = 5, M = 100, strategy = "polynomial"):
        self.alpha = alpha
        self.counter = 0
        self.M = M
        self.eps = 1 / M
        self.beta = beta
        self.kwargs = {}
        self.strategy = strategy

    def __call__(self, y_true, y_pred):
        if np.isscalar(y_pred):
            y_pred = np.ones_like(y_true) * y_pred
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth(y_true)
        grad = self.grad(y_true, y_pred)
        weights = self.weights(np.abs(grad))
        loss = -(smoothed_y * np.log(y_pred) + (1 - smoothed_y) * np.log(1 - y_pred))
        return loss * weights

    def smooth(self, y):
        mean = 0.5
        return (1 - self.alpha) * y + self.alpha * mean
    
    def weights(self, grad_norms):
        histogram = np.zeros(int(self.M))
        indices = np.floor(grad_norms/self.eps).astype(int)
        histogram = np.bincount(indices)
        indices = np.floor(grad_norms / self.eps).astype(int)
        if self.strategy == "polynomial":
            weights = (self.eps / histogram[indices]) ** (1 / self.beta)
        else:
            weights = (self.eps / histogram[indices])
            weights = np.log(weights)
            weights -= (np.min(np.log(weights)) - 1e-3)
            weights = weights ** self.beta
        return weights

    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth(y_true)
        gradient = y_pred - smoothed_y
        res = gradient.reshape(y_true.shape)

        return res

    def hess(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        hessian = y_pred * (1 - y_pred)
        return np.ones_like(y_true) * 1e-10

    def init_score(self, y_true):
        p = np.clip(np.mean(y_true), 1e-15,  1 - 1e-15)
        log_odds = np.log(p / (1 - p))
        return log_odds

    def obj(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        grad, hess = self.grad(y, p), self.hess(y, p)
        grad_norms = np.abs(grad)
        weights = self.weights(grad_norms)

        return weights * grad, hess
    def adapt_hyperparameters(self, X, y):
        pass

    def eval(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        is_higher_better = False
        return ('gradient_harmonized_loss', self(y, p).mean(), is_higher_better)

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key == "M":
                    self.eps = 1 / self.M
            else:
                self.kwargs[key] = value
        return self

    def parameter_grid(self):
        return {
            'beta': ("suggest_loguniform", 50, 500),
            'M': ("suggest_loguniform", 10, 10000)
        }
