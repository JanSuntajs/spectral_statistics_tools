"""
A module with functions which
maninly serve as helper routines.
"""

import numpy as np
import pandas as pd


def running_mean(x, N):
    """
        Perform a running mean
        with a window of length N
        on a set of data x.
    """

    cumsum = np.cumsum(np.insert(x, 0, 0))

    return (cumsum[N:] - cumsum[:-N]) / float(N)


def running_stats(x, N):
    """
            Perform a running mean and
            standard deviation
            calculation with a window
            of length N and a set of
            data x

    """

    x = pd.Series(x)

    x = x.rolling(window=N, center=True)

    mean = x.mean().values
    mean = mean[~np.isnan(mean)]
    std = x.std().values
    std = std[~np.isnan(std)]
    return mean, std

