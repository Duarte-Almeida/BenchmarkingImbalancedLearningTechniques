import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest


class CustomIsolationForest(IsolationForest):
    def predict(self, X):
        anomaly_labels = super().predict(X)
        
        binary_labels = np.where(anomaly_labels == -1, 1, 0)

        return binary_labels
    
    def predict_proba(self, X):
        anomaly_scores = self.decision_function(X)

        min_score, max_score = min(anomaly_scores), max(anomaly_scores)
        rescaled_scores = (anomaly_scores - min_score) / (max_score - min_score)

        prob_positive = 1 - rescaled_scores
        prob_negative = rescaled_scores

        return np.column_stack((prob_negative, prob_positive))

def meta_learner_auc():
    return lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42)

def meta_learner_fpr():
    return CustomIsolationForest(contamination=0.15, random_state=42)


metalearner_names = {
    'AUC':meta_learner_auc,
    'FPR':meta_learner_fpr
}

def fetch_metalearner(name):
    assert name in metalearner_names
    return metalearner_names[name]()