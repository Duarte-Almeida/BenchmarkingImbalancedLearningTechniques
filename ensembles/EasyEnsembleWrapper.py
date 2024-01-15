from imblearn.ensemble import EasyEnsembleClassifier
import numpy as np

class EasyEnsembleWrapper(EasyEnsembleClassifier):
    def __init__(self, estimator=None, n_estimators=10, replacement=False, random_state=None, **kwargs):
        super().__init__(estimator=estimator, n_estimators=n_estimators, replacement=replacement, random_state=random_state, **kwargs)

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else:
            threshold = 0.5
        all_preds = np.zeros((X.shape[0], len(self.estimators_)))

        for i, estimator in enumerate(self.estimators_):
            preds = estimator.predict(X)
            all_preds[:, i] = preds

        aggregated_preds = np.mean(all_preds, axis=1)

        y_pred = (aggregated_preds > threshold).astype(int)
        return y_pred

    def predict_proba(self, X):
        all_probs = np.zeros((X.shape[0], 2))

        for estimator in self.estimators_:
            probs = estimator.predict_proba(X)
            all_probs += probs

        avg_probs = all_probs / len(self.estimators_)
        return avg_probs
