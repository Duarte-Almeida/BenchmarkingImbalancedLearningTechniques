import numpy as np
from scipy.optimize import minimize_scalar
import sklearn.metrics


def compute_J_cutoff(y_test, y_pred):
    fpr, tpr, th = sklearn.metrics.roc_curve(y_test, y_pred)
    J_statistic = tpr + 1 - fpr
    return th[np.argmax(J_statistic)]

def compute_FPR_cutoff(y_test, y_pred):
    fpr, tpr, th = sklearn.metrics.roc_curve(y_test, y_pred)
    fpr_th = 0.05 # default fpr value
    max_fpr_idx = np.argmax(fpr[fpr < fpr_th])
    res = th[max_fpr_idx]
    return res


def compute_cs_threshold(y_test, y_pred):
    """
    Compute the optimal probability threshold using a cost-sensitive approach with optimization.

    Parameters:
    - y_test: true labels (0 or 1)
    - y_pred: predicted probabilities from the estimator

    Returns:
    - optimal_threshold: the optimal probability threshold
    - best_cost_matrix: the best cost matrix
    """
    best_cost = float('inf')
    best_cost_matrix = None
    imbalance_ratio = sum(y_test == 0) / sum(y_test == 1)

    # Costs: [[TP, FP], [FN, TN]]
    cost_matrices_to_try = [
        [[0, 1], [imbalance_ratio, 0]], # Seems fine
    ]

    def cost_function(threshold, y_test, y_pred, cost_matrix):
        y_pred_binary = (y_pred > threshold).astype(int)
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred_binary)
        total_cost = sum(sum(cm * cost_matrix))
        return total_cost

    for cost_matrix in cost_matrices_to_try:
        result = minimize_scalar(lambda t: cost_function(t, y_test, y_pred, cost_matrix), bounds=(0, 1), method='bounded')
        threshold = result.x
        total_cost = result.fun

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
            best_cost_matrix = cost_matrix

    print(f"Best Cost Matrix = {best_cost_matrix}")
    return best_threshold


threshold_names = {
    'FPR':compute_FPR_cutoff, 
    'J':compute_J_cutoff,
    'CS':compute_cs_threshold
}

def fetch_threshold(name, y_test, y_pred):
    assert name in threshold_names
    return threshold_names[name](y_test, y_pred)
