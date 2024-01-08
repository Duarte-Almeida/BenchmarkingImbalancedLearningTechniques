import numpy as np
from scipy import optimize
from scipy import special
import sklearn.metrics
from skopt import space
import matplotlib.pyplot as plt
from scipy.stats import uniform


class CrossEntropyLoss:
    '''
    A class implementing the corresponding loss of Label Smoothing preprocessing
    adapted from: https://maxhalford.github.io/blog/lightgbm-focal-loss/ 
    '''
    def __init__(self, alpha=0, smooth=False, weight = 1):
        self.alpha = alpha
        self.smooth = smooth
        self.counter = 0
        self.kwargs = {}
        self.weight = weight
        self.grid = {}
        if self.smooth:
            self.grid["alpha"] = ("suggest_uniform", 0.0, 0.5)
        self.grid["weight"] = ("suggest_uniform", 1.0, 1000.0 - 1.0)

    def __call__(self, y_true, y_pred):
        if np.isscalar(y_pred):
            y_pred = np.ones_like(y_true) * y_pred
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth_targets(y_true)
        loss = -(self.weight * smoothed_y * np.log(y_pred) + (1 - smoothed_y) * np.log(1 - y_pred))
        return loss

    def smooth_targets(self, y):
        mean = 0.5
        return (1 - self.alpha) * y + self.alpha * mean

    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        smoothed_y = self.smooth_targets(y_true)
        gradient = y_pred - smoothed_y
        gradient[y_true == 1] *= self.weight
        res = gradient.reshape(y_true.shape)

        return res

    def hess(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15,  1 - 1e-15)
        hessian = y_pred * (1 - y_pred)
        hessian = 1e-10 * np.ones_like(y_true)
        hessian[y_true == 1] *= self.weight
        return hessian.reshape(y_true.shape)

    def init_score(self, y_true):
        p = np.clip(np.mean(y_true), 1e-15,  1 - 1e-15)
        log_odds = np.log(p / (1 - p))
        return log_odds

    def obj(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        grad = self.grad(y, p)
        #print(f"Will use this weight: {self.weight}")
        return (
         self.grad(y, p), self.hess(y, p))

    def eval(self, train_data, preds):
        y = train_data
        p = special.expit(preds)
        is_higher_better = False
        #self.counter += 1
        #if self.counter % 10 == 0:
        #    fpr, tpr, th = sklearn.metrics.roc_curve(train_data, p)
        #    fpr_th = 0.05 
        #    max_fpr_idx = np.argmax(fpr[fpr < fpr_th])
        #    res_th = th[max_fpr_idx]
        #    M = 100
        #    eps = 1 / M
        #    C_histogram = np.zeros(M)
        #    A_histogram = np.zeros(M)
        #    counts = np.zeros(M)
        #    for (pred, true) in zip(p, y):
        #        y_pred = pred > res_th
        #        conf = pred * y_pred + (1 - pred) * (1 - y_pred)
        #        idx = min(int(np.floor(conf / eps)), M - 1)
        #        counts[idx] += 1
        #        C_histogram[idx] += conf
        #        A_histogram[idx] += (y_pred == true)
        #    
        #    C_histogram[C_histogram != 0] = C_histogram[C_histogram != 0] / counts[C_histogram != 0]
        #    A_histogram[A_histogram != 0] = A_histogram[A_histogram != 0] / counts[A_histogram != 0]
        #    histogram = C_histogram - A_histogram
        #    plt.bar([i for i in range(1, M + 1)], histogram)
        #    plt.show()
        return ('cross_entropy_loss', self(y, p).mean(), is_higher_better)
    
    def adapt_hyperparameters(self, X, y):
        IR = y[y == 0].shape[0] / y[y == 1].shape[0]
        #print(f"Maximum is now {IR}")
        self.grid["weight"] = ("suggest_uniform", 1.0, max(float(np.sqrt(IR)) - 1.0, 1 + 1e-5))
        #print(f"This is my fucking grid now: {self.grid}")

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
