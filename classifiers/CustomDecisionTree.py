import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class CustomDecisionTree(BaseEstimator):
    def __init__(self, loss_fn, max_depth=None, random_state=None):
        self.loss_fn = loss_fn
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)

        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        # Check termination conditions
        if depth == self.max_depth or len(np.unique(y)) == 1:
            leaf_value = self.loss_fn.compute_leaf_value(y)
            return {'leaf': True, 'value': leaf_value}

        # Find the best split
        split_index, split_value = self._find_best_split(X, y)

        if split_index is None:
            # No suitable split found, create a leaf node
            leaf_value = self.loss_fn.compute_leaf_value(y)
            return {'leaf': True, 'value': leaf_value}

        # Split the data
        left_mask = X[:, split_index] <= split_value
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return the node representing the split
        return {
            'leaf': False,
            'split_index': split_index,
            'split_value': split_value,
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, y):
        # Implement your custom split finding logic based on the loss function
        # This might involve iterating through features and values to find the best split

        # Sample implementation: find split using your custom loss function
        best_split_index = None
        best_split_value = None
        best_score = np.inf

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            for value in unique_values:
                left_mask = X[:, feature_index] <= value
                right_mask = ~left_mask

                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    # Customize this line to use your custom loss function
                    score = self.loss_fn(y[left_mask], y[right_mask])

                    # Adjust the comparison based on your optimization goal
                    if score < best_score:
                        best_score = score
                        best_split_index = feature_index
                        best_split_value = value

        return best_split_index, best_split_value

    def predict(self, X):
        check_is_fitted(self, 'tree')
        X = check_array(X)
        return np.array([self._predict_tree(self.tree_, x) for x in X])

    def _predict_tree(self, node, x):
        if node['leaf']:
            return node['value']

        if x[node['split_index']] <= node['split_value']:
            return self._predict_tree(node['left'], x)
        else:
            return self._predict_tree(node['right'], x)

    def predict_proba(self, X):
        check_is_fitted(self, 'tree')
        X = check_array(X)
        return np.array([self._predict_proba_tree(self.tree_, x) for x in X])

    def _predict_proba_tree(self, node, x):
        if node['leaf']:
            return self.loss_fn.compute_leaf_proba(node['value'])

        if x[node['split_index']] <= node['split_value']:
            return self._predict_proba_tree(node['left'], x)
        else:
            return self._predict_proba_tree(node['right'], x)