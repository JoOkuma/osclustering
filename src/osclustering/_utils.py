import logging

import higra as hg
import numpy as np

LOG = logging.getLogger(__name__)


def _validate_tree(tree: hg.Tree | np.ndarray) -> hg.Tree:
    """
    Validates a tree and converts it to a higra tree if it is a numpy array.

    Parameters
    ----------
    tree : hg.Tree | np.ndarray
        Tree to validate.

    Returns
    -------
    hg.Tree
        Validated tree.
    """
    if isinstance(tree, np.ndarray):
        tree = hg.scipy_linkage_matrix_to_binary_hierarchy(tree)

    return tree


_COLORS = np.array(
    [
        "#000000",
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#a65628",
        "#f781bf",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
)


def to_colors(y: np.ndarray) -> np.ndarray:
    if y.max() > len(_COLORS):
        LOG.warning("More clusters than colors, some colors will be repeated.")
    return _COLORS[y % len(_COLORS)]
