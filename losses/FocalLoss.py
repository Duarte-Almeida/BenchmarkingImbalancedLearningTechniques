import numpy as np
from scipy import optimize
from scipy import special
from skopt import space
import sklearn.metrics
import matplotlib.pyplot as plt
from scipy.stats import uniform

class FocalLoss:
    ''' source: https://maxhalford.github.io/blog/lightgbm-focal-loss/
    ''' 

    def __init__(self, gamma = 1, alpha=None):
        self.alpha = alpha
        self.gamma = gamma
        self.kwargs = {}
        self.count = 0
        self.grid = {
            'gamma': ("suggest_uniform", 0.0, 2.0),
            'alpha': ("suggest_uniform", 1.0, 1000.0)
        }

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y = 2 * y_true - 1
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        #y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        #y = 2 * y_true - 1
        #at = self.at(y_true)
        #pt = self.pt(y_true, y_pred)
        #g = self.gamma
        #u = at * y * (1 - pt) ** g
        #du = -at * y * g * (1 - pt) ** (g - 1)
        #v = g * pt * np.log(pt) + pt - 1
        #dv = g * np.log(pt) + g + 1
        #hess = (du * v + u * dv) * y * (pt * (1 - pt))
        #hess[hess < 0] = 0
        hess = np.ones_like(y_true) * 1e-10
        return hess

    def init_score(self, y_true):
        p = np.clip(np.mean(y_true), 1e-15, 1 - 1e-15)
        log_odds = np.log(p / (1 - p))
        return log_odds

    def obj(self, train_data, preds):
        #print(preds)
        y = train_data
        p = special.expit(preds)
        #print(self.grad(y, p).mean())
        #print(self.hess(y, p).mean())
        return (self.grad(y, p), self.hess(y, p))

    def eval(self, train_data, preds):
        y = train_data
        p = special.expit(preds)

        is_higher_better = False
        return ('focal_loss', self(y, p).mean(), is_higher_better)

    def adapt_hyperparameters(self, X, y):
        IR = y[y == 0].shape[0] / y[y == 1].shape[0]
        self.grid["alpha"] = ("suggest_uniform", 1.0, max(1, float(np.sqrt(IR))))
        #pass
    
    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self

    def parameter_grid(self):
        return self.grid
