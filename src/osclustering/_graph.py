import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsTransformer, kneighbors_graph


class KMSTNeighborsTransformer(KNeighborsTransformer):
    """
    A transformer that computes the k-nearest neighbors graph using the minimum spanning tree
    to guarantee connectedness.
    """

    def __init__(self, n_neighbors: int, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y=None):
        raise NotImplementedError("This class is not meant to be used directly.")

    def transform(self, X):
        raise NotImplementedError("This class is not meant to be used directly.")

    def fit_transform(self, X, y=None):
        distances = knn_mst_graph(X, **self.get_params())
        return distances


def knn_mst_graph(
    X: np.ndarray,
    n_neighbors: int,
    **kwargs,
) -> sparse.csr_matrix:
    """
    Builds a symmetric k-nearest neighbors graph using the minimum spanning tree
    to guarantee connectedness.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    n_neighbors : int
        Number of neighbors.
    **kwargs : dict
        Keyword arguments for `sklearn.neighbors.kneighbors_graph`.

    Returns
    -------
    sparse.csr_matrix
        k-nearest neighbors graph.
    """
    A = kneighbors_graph(X, n_neighbors, mode="distance", **kwargs)
    A = A + A.T
    D = squareform(pdist(X))
    MST = sparse.csgraph.minimum_spanning_tree(D)
    MST = MST + MST.T
    A = A.maximum(MST)
    return sparse.csr_matrix(A)
