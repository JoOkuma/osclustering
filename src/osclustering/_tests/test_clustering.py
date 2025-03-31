import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from osclustering import (
    knn_mst_graph,
    optimal_stability_clustering,
    trees_from_edge_ranking_perturbation,
)
from osclustering._data import sklearn_datasets
from osclustering._utils import to_colors

try:
    import matplotlib.pyplot as plt

except ImportError:
    plt = None


@pytest.mark.parametrize(
    "exact",
    [False, True],
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
        reference_tree, perturbated_trees = trees_from_edge_ranking_perturbation(
            graph,
            n_replicates,
            max_displacement,
        )

        Yhat = optimal_stability_clustering(
            reference_tree,
            perturbated_trees,
            exact=exact,
        )

        score = adjusted_rand_score(Y, Yhat)

        if request.config.getoption("--plot"):
            if plt is None:
                raise ImportError("matplotlib is required to plot the results")

            plt.scatter(X[:, 0], X[:, 1], c=to_colors(Yhat))
            plt.show()

        n_gt_clusters = len(np.unique(Y))
        n_pred_clusters = len(np.unique(Yhat))

        assert n_gt_clusters == n_pred_clusters
        assert score > 0.8
