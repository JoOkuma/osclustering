import logging

import numpy as np
import scanpy as sc

LOG = logging.getLogger(__name__)


def dropout_counts(
    adata: sc.AnnData,
    dropout_rate: float = 0.2,
    n_replicates: int = 10,
    random_seed: int = 0,
) -> list[sc.AnnData]:
    """
    Generate trees from dropout data.
    """
    if not 0 <= dropout_rate <= 1:
        raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

    rng = np.random.default_rng(random_seed)

    if "counts" not in adata.layers:
        LOG.warning("No counts layer found in adata. Using X layer.")
        X = adata.X
    else:
        X = adata.layers["counts"]

    try:
        X = X.toarray()
    except AttributeError:
        pass

    if not np.issubdtype(X.dtype, np.integer):
        LOG.warning("X is not an integer array. Converting to int.")
        X = X.astype(int)

    trees = []
    for _ in range(n_replicates):
        adata_dropout = adata.copy()
        adata_dropout.X = rng.binomial(
            n=X,
            p=1.0 - dropout_rate,
        )
        trees.append(adata_dropout)

    return trees
