from self_paced_ensemble import SelfPacedEnsembleClassifier
import numpy as np
import classifiers
import sklearn.base

class SelfPacedEnsembleWrapper(SelfPacedEnsembleClassifier):
    def __init__(self, base_estimator=classifiers.LGBM(), n_estimators=10, replacement=False, random_state=42, **kwargs):
        self.grid = {
            "n_estimators" : ("suggest_int", 5, 20),
            "k_bins": ("suggest_int", 5, 100),
        }
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators, replacement=replacement, random_state=random_state, **kwargs)


    def fit(self, X, y, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)

    def predict(self, X, **kwargs):
        if "th" in kwargs:
            threshold = kwargs["th"]
        else:
            threshold = 0.5
        all_preds = np.zeros((X.shape[0], len(self.estimators_)))

        for i, estimator in enumerate(self.estimators_):
            preds = estimator.predict_proba(X)[:, 1]
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

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._get_param_names()}
    
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
                    if attr not in attr_params:
                        attr_params[attr] = {}
                    attr_params[attr][attr_param] = value
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        for attr, attr_dict in attr_params.items():
            getattr(self, attr).set_params(**attr_dict)

        return self

    def parameter_grid(self):
        return self.grid
    
    def adapt_hyperparameters(self, X, y):
        self.base_estimator.adapt_hyperparameters(X, y)
