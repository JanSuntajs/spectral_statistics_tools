"""
A module with functions
used in filtering the
data for SFF calculation.

All the functions have the
following common interface:

filter_func(spectra, *args, **kwargs)

Params:
-------

spectra: ndarray
            A 2D ndarray.
*args and **kwargs are additional
positional and keyword arguments.

"""

import numpy as _np
# import decorators as _dec
from utils import tester_methods as _tst
import functools


def check_dims(func):
    @functools.wraps(func)
    def wrapper_check_dims(*args, **kwargs):

        args = list(args)
        args[0] = _tst._check_spectral_dimensions(args[0], 2)
        args = tuple(args)
        value = func(*args, **kwargs)

        return value
    return wrapper_check_dims


@check_dims
def __identity(spectra, *args, **kwargs):
    """
    No filtering is used in the SFF calculation.
    Parameters:

    Parameters:
    -----------
    spectra: ndarray
                2D ndarray of energy spectra.


    Returns
    -------
    filter_: ndarray
                2D ndarray, dtype=float, has the same shape as spectra,
                filled with ones.

    """

    filter_ = _np.ones_like(spectra, dtype=_np.float64)

    return filter_


@check_dims
def __gaussian_filter(spectra, misc_dict, eta = 1., *args, **kwargs):
    """
    Apply Gaussian filtering to the
    spectrum.
    Parameters:
    -----------
    spectra: ndarray
                2D ndarray of energy spectra.

    misc_dict: dict
                misc_dict of the following form:
                mean_ener: ndarray
                            1D array with nsamples values of mean energies. If
                            individual is set to true, those values are in
                            generall all different, in the opposite case they
                            are all equal to the mean taken over all the
                            spectra.
                            The same holds for all other returned quantities.
                sq_ham_tr: ndarray
                            1D array of the same shape as mean_ener. Returns
                            the value of the squared mean of the Hamiltonian's
                            trace.
                ham_tr_sq: ndarray
                            1D array of the same shape as mean_ener. Returns
                            the value of the trace of the squared Hamiltonian.
                gamma: ndarray
                            1D array of the same shape as mean_ener, returns
                            the Gamma values where Gamma was defined above.

                unfolded: boolean
                            Whether True or False was chosen for the unfolded
                            parameter.
                individual: boolean
                            Whether True or False was chosen for the individual
                            parameter.
    eta: float
                eta value, the percentage of the spectrum to account
                for in filtering.
    Returns
    -------
    filter_: ndarray
                2D ndarray, dtype=float, has the same shape as spectra.
    """

    gamma = misc_dict['gamma']
    e_mean = misc_dict['mean_ener']

    e_dev = gamma * eta
    filter_ = _np.exp(-0.5 * (spectra -
                              e_mean[:, None]) ** 2 / e_dev[:, None] ** 2)

    return filter_


available_filters = {

    'identity': __identity,
    'gaussian': __gaussian_filter

}
