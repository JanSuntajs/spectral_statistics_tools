"""
A module with functions which
maninly serve as helper routines.
"""

import numpy as np


def running_mean(x, N):
    """
        Perform a running mean
        with a window of length N
        on a set of data x.
    """

    cumsum = np.cumsum(np.insert(x, 0, 0))

    return (cumsum[N:] - cumsum[:-N]) / float(N)
