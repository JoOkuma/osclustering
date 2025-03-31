from functools import partial
from typing import Literal

import higra as hg
import numpy as np
from scipy import sparse

from osclustering._multiprocessing import _multiprocessing_apply

HIER_TYPE_TO_FUNC = {
    "mst": hg.bpt_canonical,
    "area": partial(hg.watershed_hierarchy_by_area, canonize_tree=False),
    "parents": partial(
        hg.watershed_hierarchy_by_number_of_parents, canonize_tree=False
    ),
}


def trees_with_edge_ranking_perturbation(
    graph: sparse.csr_array,
    n_replicates: int,
    max_edge_displacement: int,
    linkage_mode: Literal["mst", "area", "parents"] = "parents",
    random_seed: int = 0,
    n_jobs: int = 1,
) -> tuple[hg.Tree, list[hg.Tree]]:
    """Compute perturbed trees from graph and weights.

    Parameters
    ----------
    graph : sparse.csr_array
        Graph.
    n_replicates : int
        Number of replicates.
    max_edge_displacement : int
        Maximum edge displacement.
    random_seed : int
        Random seed.
    n_jobs : int
        Number of jobs.

    Returns
    -------
    tuple[hg.Tree, list[hg.Tree]]
        Reference tree and list of perturbed trees.
    """
    rng = np.random.default_rng(random_seed)
    hg_graph, hg_weights = hg.adjacency_matrix_2_undirected_graph(graph)

    hier_func = HIER_TYPE_TO_FUNC[linkage_mode]

    ref_tree, _ = hier_func(hg_graph, hg_weights)
    trees = [ref_tree]
    rank_weights = hg.arg_sort(hg.arg_sort(hg_weights, stable=True), stable=True)
    perturbation = rng.integers(
        0,
        max_edge_displacement + 1,
        size=(n_replicates, len(rank_weights)),
    )

    trees = _multiprocessing_apply(
        lambda i: hier_func(
            hg_graph,
            rank_weights + perturbation[i],
        )[0],
        range(n_replicates),
        n_jobs=n_jobs,
    )

    return ref_tree, trees
