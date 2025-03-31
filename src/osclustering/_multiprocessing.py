import multiprocessing as mp
from typing import Any, Callable, List, Sequence

import cloudpickle
from multiprocessing.reduction import ForkingPickler

"""
Configures multiprocessing to use multiprocessing for pickling.
This allows function to be pickled.
Reference: https://stackoverflow.com/a/69253561/6748803
"""
# Update the default reducer for multiprocessing
ForkingPickler.dumps = cloudpickle.dumps
ForkingPickler.loads = cloudpickle.loads

def _multiprocessing_apply(
    func: Callable[[Any], None],
    sequence: Sequence[Any],
    n_jobs: int,
) -> List[Any]:
    """Applies `func` for each item in `sequence`.

    Parameters
    ----------
    func : Callable[[Any], NoneType]
        Function to be executed.
    sequence : Sequence[Any]
        Sequence of parameters.
    n_jobs : int
        Number of jobs for multiprocessing.

    Returns
    -------
    List[int]
        List of `func` outputs.
    """
    length = len(sequence)
    if n_jobs > 1 and length > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(n_jobs, length)) as pool:
            return pool.imap(func, sequence)
    else:
        return [func(t) for t in sequence]
