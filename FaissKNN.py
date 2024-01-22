import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin
from sklearn.utils.validation import check_array
import faiss

class FaissKNN(NeighborsBase, KNeighborsMixin):

    def __init__(self, n_neighbors=5, metric="l2"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kwargs = {}

    def fit(self, X, y=None):
        X = check_array(X)
        self.X_ = np.ascontiguousarray(X)
        self.y_ = np.ascontiguousarray(y)
        self.index_ = faiss.IndexFlatL2(self.X_.shape[1])
        self.index_.add(self.X_.astype(np.float32))
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            X = self.X_
        else:
            X = check_array(X)
            X = np.ascontiguousarray(X)

        n_neighbors = n_neighbors or self.n_neighbors

        D, I = self.index_.search(X.astype(np.float32), n_neighbors)

        if return_distance:
            return D, I
        else:
            return I

    def predict(self, X):
        D, I = self.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=True)

        # Use inverse of distances as weights for voting
        weights = 1 / (D + 1e-6)  # Adding a small epsilon to avoid division by zero

        # Perform weighted voting to predict the class
        weighted_votes = np.zeros((X.shape[0], len(np.unique(self.y_))), dtype=np.float32)
        for i in range(self.n_neighbors):
            weighted_votes += np.eye(len(np.unique(self.y_)))[self.y_[I[:, i]]] * weights[:, i][:, np.newaxis]

        predictions = np.argmax(weighted_votes, axis=1)
        
        return predictions

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        if mode not in ['connectivity', 'distance']:
            raise ValueError(f"Invalid mode '{mode}'. Use 'connectivity' or 'distance'.")

        n_neighbors = n_neighbors or self.n_neighbors
        D, I = self.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)

        n_samples = self.X_.shape[0]
        row_indices = np.repeat(np.arange(n_samples), n_neighbors)
        col_indices = I.flatten()
        if mode == 'connectivity':
            data = np.ones(n_samples * n_neighbors)
        else: 
            data = D.flatten()

        knn_graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))

        return knn_graph

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
        return {'n_neighbors': ("suggest_categorical", [i for i in range(2, 8)])}
