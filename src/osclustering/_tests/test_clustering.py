import matplotlib.pyplot as plt
import pytest

from osclustering import (
    knn_mst_graph,
    optimal_stability_clustering,
    trees_with_edge_ranking_perturbation,
)
from osclustering._data import sklearn_datasets


@pytest.mark.parametrize(
    "exact",
    [True, False],
)
def test_stability_clustering(
    exact: bool,
    request,
):
    n_samples = 500
    n_neighbors = 10
    n_replicates = 20
    max_displacement = int(n_neighbors * n_samples * 0.5)

    for X, Y in sklearn_datasets(n_samples):
        graph = knn_mst_graph(X, n_neighbors=n_neighbors)
        print("0")
        reference_tree, perturbated_trees = trees_with_edge_ranking_perturbation(
            graph,
            n_replicates,
            max_displacement,
        )
        print("1")

        Yhat = optimal_stability_clustering(
            reference_tree,
            perturbated_trees,
            exact=exact,
        )
        print("2")
        if request.config.getoption("--plot"):
            plt.scatter(X[:, 0], X[:, 1])
            plt.show()
