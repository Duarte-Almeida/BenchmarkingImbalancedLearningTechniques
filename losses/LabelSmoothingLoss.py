import numpy as np
from scipy import optimize
from scipy import special
import sklearn.metrics

class LabelSmoothingLoss:
    '''
    A class implementing the corresponding loss of Label Smoothing preprocessing
    adapted from: https://maxhalford.github.io/blog/lightgbm-focal-loss/ 
    '''
    def __init__(self, alpha=0.1, freeze=False):
        self.alpha = alpha
        self.freeze = freeze

    def __call__(self, y_true, y_pred):
        if np.isscalar(y_pred):
            y_pred = np.ones_like(y_true) * y_pred
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth(y_true)
        loss = -(smoothed_y * np.log(y_pred) + (1 - smoothed_y) * np.log(1 - y_pred))
        return loss

    def smooth(self, y):
        mean = 0.5
        return (1 - self.alpha) * y + self.alpha * mean

    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth(y_true)
        gradient = y_pred - smoothed_y
        return gradient.reshape(y_true.shape)

    def hess(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        hessian = y_pred * (1 - y_pred)
        return hessian.reshape(y_true.shape)

    def init_score(self, y_true):
        p = np.clip(np.mean(y_true), 1e-15,  1 - 1e-15)
        log_odds = np.log(p / (1 - p))
        #print(f"Starting from p = {p} and {log_odds}")
        return log_odds

    def obj(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        return (
         self.grad(y, p), self.hess(y, p))

    def eval(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        is_higher_better = False
        return ('label_smoothing_loss', self(y, p).mean(), is_higher_better)

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
        if self.freeze:
            return {}
        return {'alpha': [0.05, 0.1, 0.2, 0.5]}
