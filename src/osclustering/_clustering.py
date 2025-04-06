import logging
from collections import deque
from typing import Sequence

import gurobipy as gp
import higra as hg
import numpy as np
import scipy.sparse as sparse
from gurobipy import GRB

from osclustering._multiprocessing import _multiprocessing_apply
from osclustering._utils import _validate_tree

LOG = logging.getLogger(__name__)


def _parents_to_binary_features(parents: np.ndarray) -> sparse.csr_array:
    """
    Converts a tree to (2N - 1, N) binary representation.
    Where N is the number of leaves of the tree.
    Each row represents a node and each column a leaf.

    Parameters
    ----------
    parents : np.ndarray
        Tree representated as parent relationship array.

    Returns
    -------
    sparse.csr_array
        Binary encoding of tree.
    """
    num_leaves = (parents.shape[0] + 1) // 2
    cols = [[i] for i in range(num_leaves)] + [
        [] for _ in range(parents.shape[0] - num_leaves)
    ]

    for i in range(parents.shape[0] - 1):
        cols[parents[i]] += cols[i]

    rows = []
    for i, row in enumerate(cols):
        rows += [i] * len(row)

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.concatenate(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=bool)

    X = sparse.coo_array(
        (data, (rows, cols)),
        shape=(parents.shape[0], num_leaves),
    ).tocsr()

    return X


def _clear_leaves_and_root_weights(D: sparse.csr_array) -> sparse.csr_array:
    """
    Zeros-out the weights of leaves and root from a similarity matrix.

    Parameters
    ----------
    D : sparse.csr_array
        (2N - 1) x (2N - 1) similarity matrix.

    Returns
    -------
    sparse.csr_array
        Similarity matrix with leaves and root weights zeroed-out.
    """
    N = (D.shape[0] + 1) // 2

    # removing cols
    D = D.tocsc()
    start, end = D.indptr[N], D.indptr[-2]
    D.data[:start] = 0
    D.data[end:] = 0

    # removing rows
    D = D.tocsr()
    start, end = D.indptr[N], D.indptr[-2]
    D.data[:start] = 0
    D.data[end:] = 0

    return D


def _solution_to_labels(
    Y: np.ndarray,
    features: sparse.csr_array,
) -> np.ndarray:
    """Converts a solution to a label assignment.

    Parameters
    ----------
    Y : np.ndarray
        Problem solution indices.
    features : sparse.csr_array
        Binary encoding of tree.

    Returns
    -------
    np.ndarray
        Label assignment.
    """
    assert features.dtype == bool

    n_labels = 1
    labels = np.zeros(features.shape[1], dtype=int)

    for y in Y:
        mask = features[[y]].toarray()[0]
        labels[mask] = n_labels
        n_labels += 1

    return labels


def _exact_stability_clustering(
    trees: list[hg.Tree],
    similarities: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Solve the rank stable clustering problem with one to all relationships.

    Parameters
    ----------
    trees : list[hg.Tree]
        Sequence of N trees.
    similarities : list[np.ndarray]
        Sequence of N - 1 similarities.

    Returns
    -------
        dict[int, float]
        ILP solution by sample index and objective value.
    """
    assert len(trees) == len(similarities) + 1

    tree_size = trees[0].num_vertices()

    m = gp.Model()
    m.ModelSense = GRB.MAXIMIZE
    m.Params.OutputFlag = 0

    W = {
        (t, r, c): S[r, c]
        for t, S in enumerate(similarities)
        for r, c in zip(*S.nonzero())
    }

    X = m.addVars(W.keys(), obj=W, vtype=GRB.BINARY)
    Ys = [m.addVars(tree_size, vtype=GRB.BINARY) for _ in range(len(trees))]

    for t, Y in enumerate(Ys[1:]):
        # out-going constraints from reference nodes
        m.addConstrs(Ys[0][i] == X.sum(t, i, "*") for i in range(tree_size))
        # in-coming constraints to auxiliary nodes
        m.addConstrs(Y[i] == X.sum(t, "*", i) for i in range(tree_size))

    for Y, tree in zip(Ys, trees):
        # add ancestor constraints
        for k in Y.keys():
            m.addConstr(gp.quicksum(Y[i] for i in tree.ancestors(k)) <= 1)

    m.optimize()

    solution = np.asarray([k for k, var in Ys[0].items() if var.X > 0])

    return solution, m.objVal


def _optimal_assignment_weights(similarity: np.ndarray) -> np.ndarray:
    """
    Compute the optimal assignment weights based on similarity.

    Parameters
    ----------
    similarity : np.ndarray
        A 2D array of shape (n, n) representing the similarity scores
        between elements.

    Returns
    -------
    np.ndarray
        An array containing the optimal assignment weights based on the given similarity matrix.

    Notes
    -----
    The function uses the linear_sum_assignment algorithm, which finds the optimal assignment
    for maximum weight matching in bipartite graphs. The negative of the similarity matrix
    is passed to the function since it's a minimization algorithm.

    Raises
    ------
    AssertionError
        If the number of rows doesn't match the shape of the similarity matrix.
    """
    N = (similarity.shape[0] + 1) // 2
    # ignoring leaves and roots
    row, col = sparse.csgraph.min_weight_full_bipartite_matching(
        -similarity[N:-1, N:-1]
    )
    assert row.shape[0] == similarity.shape[0] - N - 1

    weights = np.zeros(similarity.shape[0], dtype=float)
    weights[N:-1] = similarity[N + row, N + col]
    return weights


def _solve_maximum_with_ancestor_constraints(
    weights: np.ndarray,
    tree: hg.Tree,
) -> tuple[np.ndarray, float]:
    """
    Solve the maximum problem considering ancestor constraints.

    Parameters
    ----------
    weights : np.ndarray
        1D array containing weights for each node in the tree.
    tree : hg.Tree
        A Tree object representing the hierarchical structure.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing:
        - An array with the optimal solution nodes.
        - A float value representing the maximum weight of the solution.

    Notes
    -----
    The function works by iterating through the tree hierarchy
    and combining solutions of child nodes to get the optimal solution
    for parent nodes.
    """
    tree_size = tree.num_vertices()
    graph_size = tree.num_leaves()
    num_children = hg.attribute_area(tree)

    solutions = [[i] for i in range(graph_size)] + [
        [] for _ in range(tree_size - graph_size)
    ]

    parents = tree.parents()
    count = np.zeros(tree_size, dtype=int)
    count[:graph_size] = 1

    left_children = tree.child(0)
    right_children = tree.child(1)

    accum_weights = np.zeros(tree_size, dtype=float)
    accum_weights[:graph_size] = weights[:graph_size]

    Q = deque(range(graph_size), maxlen=graph_size)
    while Q:
        i = Q.popleft()
        p = parents[i]

        # if not leaf
        if i >= graph_size:
            left_child = left_children[i - graph_size]
            right_child = right_children[i - graph_size]
            children_weights = accum_weights[left_child] + accum_weights[right_child]
            if children_weights > weights[i]:
                accum_weights[i] = children_weights
                solutions[i] = solutions[left_child] + solutions[right_child]
            else:
                accum_weights[i] = weights[i]
                solutions[i] = [i]

        count[p] += count[i]
        if count[p] == num_children[p] and p != i:
            Q.append(p)

    solution = solutions[-1]
    obj_weight = accum_weights[-1]
    return solution, obj_weight


def _approximate_stability_clustering(
    trees: list[hg.Tree],
    similarities: list[np.ndarray],
    n_jobs: int,
) -> tuple[np.ndarray, float]:
    """
    Solve the rank stable clustering problem with one to all relationships
    using approximation with optimal assigning and dynamic programming

    Parameters
    ----------
    trees : list[hg.Tree]
        Sequence of N trees.
    similarities : list[np.ndarray]
        Sequence of N - 1 similarities.
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
        dict[int, float]
        ILP solution by sample index and objective value.
    """
    assert len(trees) == len(similarities) + 1

    tree_size = trees[0].num_vertices()

    weights = np.zeros(tree_size)

    for node_weights in _multiprocessing_apply(
        _optimal_assignment_weights,
        similarities,
        n_jobs,
    ):
        weights += node_weights

    return _solve_maximum_with_ancestor_constraints(weights, trees[0])


def optimal_stability_clustering(
    reference_tree: hg.Tree | np.ndarray,
    perturbated_trees: Sequence[hg.Tree | np.ndarray],
    exact: bool = False,
    single_cluster_threshold: float = 0.9,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Perform optimal stability clustering.

    Parameters
    ----------
    reference_tree : hg.Tree | np.ndarray
        Reference tree, final clustering will be computed with respect to this tree.
    perturbated_trees : Sequence[hg.Tree | np.ndarray]
        Sequence of perturbated trees.
    exact : bool, default=False
        If True, use exact stability clustering.
    single_cluster_threshold : float, default=0.9
        If the objective value is lower than this threshold, return a single cluster.
        Set to 0 to disable this behavior.
    n_jobs : int, default=1
        Number of jobs to run in parallel.

    Returns
    -------
    np.ndarray
        Final clustering.
    """
    reference_tree = _validate_tree(reference_tree)
    perturbated_trees = [_validate_tree(tree) for tree in perturbated_trees]

    reference_features = _parents_to_binary_features(reference_tree.parents())
    similarities = _multiprocessing_apply(
        lambda x: _clear_leaves_and_root_weights(
            reference_features @ _parents_to_binary_features(x.parents()).T.astype(int)
        ),
        perturbated_trees,
        n_jobs,
    )

    trees = [reference_tree] + perturbated_trees

    if exact:
        Y, obj = _exact_stability_clustering(trees, similarities)
    else:
        Y, obj = _approximate_stability_clustering(trees, similarities, n_jobs)

    n_total_nodes = sum(tree.num_leaves() for tree in perturbated_trees)
    obj = obj / n_total_nodes

    print(f"Cluster stability score: {obj:.4f}")

    if obj <= single_cluster_threshold:
        LOG.info("Stability score is too low (%f), returning single cluster", obj)
        return np.ones(reference_features.shape[1], dtype=int)

    labels = _solution_to_labels(Y, reference_features)

    return labels
