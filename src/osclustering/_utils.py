import higra as hg
import numpy as np


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
