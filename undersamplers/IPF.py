import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import classifiers
from skopt import space

class IPFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cls = "majority", estimator=None, k=5, p=0.1):

        if estimator is None:
            self.estimator = classifiers.LGBM()
        else:
            self.estimator = estimator

        self.k = k
        self.p = p

        self.n_splits = 5
        self.removed_indices = []
        self.cls = cls

    def fit_resample(self, X, y):
        X_init_size = X.shape[0]
        n_samples_removed = np.inf
        n_iter = 0
        initial_map = np.arange(X.shape[0])
        removed = []

        #print(f"Performing Undersampling")
        while n_iter < self.k and n_samples_removed > self.p * len(X):
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            predictions = []
            thresholds = []
            
            # Get estimator performance across folds
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.estimator.fit(X_train, y_train)
                predictions.append(self.estimator.predict_proba(X)[:, 1])
                thresholds.append(self.compute_threshold(predictions[-1], y))
            
            # At least half of classifiers should agree
            pred_votes = (np.mean(predictions, axis=0) > np.mean(thresholds)).astype(int)
            if self.cls == "majority":
                to_remove = np.where((np.not_equal(pred_votes, y)) & (y == 0))[0]
            else:
                to_remove = np.where(np.not_equal(pred_votes, y))[0]

            X, y = np.delete(X, to_remove, axis=0), np.delete(y, to_remove)
            removed.extend(to_remove)
            initial_map = np.delete(initial_map, to_remove)
            if n_samples_removed != np.inf:
                n_samples_removed += len(to_remove)
            else:
                n_samples_removed = len(to_remove)
            #print(f"Removed {n_samples_removed} samples")

            n_iter += 1
        
        self.sample_indices_ = np.array([x for x in removed if x < X_init_size])
        self.resampled_idx_ = np.argmax(initial_map >= X_init_size)

        return X, y

    def compute_threshold(self, predictions, y_true):
        fpr, tpr, th = roc_curve(y_true, predictions)
        fpr_th = 0.05 # default fpr value
        max_fpr_idx = np.argmax(fpr[fpr < fpr_th])
        res = th[max_fpr_idx]
        return res

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self

    def parameter_grid(self):
        return {
            'k': ("suggest_categorical", [i for i in range(3, 5)]),
            'p': ("suggest_uniform", 0.05, 0.2)
        }