import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import classifiers
import losses

class SMOTEIPFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, oversampler=None, estimator=None, k=5, p=0.1, voting = "majority"):

        # TODO: temporary solution, just to test it out on the best configuration I got
        if oversampler is None:
            self.oversampler = SMOTE(k_neighbors=3, sampling_strategy=0.15)
        else:
            self.oversampler = oversampler

        if estimator is None:
            ls = losses.LabelSmoothingLoss(0, freeze = True)
            self.estimator = classifiers.LGBM(ls)
        else:
            self.estimator = estimator

        self.k = k
        self.p = p

        self.n_splits = 5
        self.voting = voting
        self.removed_indices = []

    def fit_resample(self, X, y):
        X_init_size = X.shape[0]
        print(f"Performing Oversampling")
        X_resampled, y_resampled = self.oversampler.fit_resample(X, y)
        n_samples_removed = np.inf
        n_iter = 0
        initial_map = np.arange(X_resampled.shape[0])
        removed = []

        print(f"Performing Undersampling")
        while n_iter < self.k and n_samples_removed > self.p * len(X):
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            predictions = []
            thresholds = []
            
            # Get estimator performance across folds
            for train_index, test_index in kf.split(X_resampled, y_resampled):
                X_train, X_test = X_resampled[train_index], X_resampled[test_index]
                y_train, y_test = y_resampled[train_index], y_resampled[test_index]

                self.estimator.fit(X_train, y_train)
                predictions.append(self.estimator.predict_proba(X_resampled)[:, 1])
                thresholds.append(self.compute_threshold(predictions[-1], y_resampled))
            
            # At least half of classifiers should agree
            if self.voting == 'majority':
                pred_votes = (np.mean(predictions, axis=0) > np.mean(thresholds)).astype(int)
                to_remove = np.where(np.not_equal(pred_votes, y_resampled))[0]
            # All classifiers should get prediction wrong
            elif self.voting == 'consensus':
                pred_votes = (np.mean(predictions, axis=0) > np.mean(thresholds)).astype(int)
                sum_votes = np.sum(predictions, axis=0)
                to_remove = np.where(np.logical_and(
                    np.not_equal(pred_votes, y_resampled),
                    np.equal(sum_votes, self.n_splits)))[0]

            X_resampled, y_resampled = np.delete(X_resampled, to_remove, axis=0), np.delete(y_resampled, to_remove)
            removed.extend(to_remove)
            initial_map = np.delete(initial_map, to_remove)
            if n_samples_removed != np.inf:
                n_samples_removed += len(to_remove)
            else:
                n_samples_removed = len(to_remove)
            print(f"Removed {n_samples_removed} samples")

            n_iter += 1
        
        self.sampled_indices_ = np.array([x for x in removed if x < X_init_size])
        self.resampled_idx_ = np.argmax(initial_map >= X_init_size)

        return X_resampled, y_resampled

    def compute_threshold(self, predictions, y_true):
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        j_statistic = tpr - fpr
        best_threshold_index = np.argmax(j_statistic)
        best_threshold = thresholds[best_threshold_index]
        return best_threshold

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self

    def parameter_grid(self):
        return {
            'k': [5],
            'p': [0.05, 0.1, 0.2], 
            'voting': ['majority']
        }