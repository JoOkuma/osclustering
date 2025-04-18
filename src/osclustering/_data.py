import numpy as np
from sklearn import datasets


def sklearn_datasets(
    n_samples: int,
    random_seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate sklearn clustering datasets.
    Their examples used `n_samples=500`.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    random_seed : int
        Random seed.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of datasets.
    """
    np.random.seed(random_seed)

    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), np.ones(n_samples, dtype=int)

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    return [noisy_circles, noisy_moons, blobs, no_structure, aniso, varied]
