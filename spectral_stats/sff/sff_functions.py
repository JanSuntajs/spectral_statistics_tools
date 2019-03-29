"""
A module with routines for calculating
spectral form factor.
sff_single() function calculates sff
for a single spectrum,
sff_spectra calculates sff for an
ensemble of spectra by repeatedly calling
the sff_single routine. numba tools are
used for speeding the calculations up.
K_GOE() function returns the theoretical
curve for normalized SFF in the GOE case.
"""

import numpy as np
import numba as nb
import pandas as pd

from ..utils import helper_functions as hlp

#  sff functions

#  using numba for speed


@nb.njit('complex128[:](float64[:], float64[:], float64[:])', fastmath=True)
def sff_single(spectrum, taulist, weights):
    """
    A function that calculates the spectral form factor
    of a single energy spectrum.

    Parameters
    ----------
    spectrum: ndarray
                A 1D ndarray of dtype np.float64 that stores the energy
                values.
    taulist: ndarray
                A 1D array that stores the values in which the spectral
                form factor should be calculated.
    weigths: ndarray
                A 1D array of the same lengths as spectrum array. Allows
                for filtering of the spectrum during the spectral form
                factor calculation.
    Returns
    -------

    sfflist: ndarray
                A 1D ndarray of dtype np.complex128 and of the same length
                as taulist array.

    """

    sfflist = np.zeros_like(taulist, dtype=np.complex128)

    for i in range(len(taulist)):
        sfflist[i] = np.sum(weights * np.exp(-1j * taulist[i] * spectrum))

    return sfflist


@nb.njit('complex128[:,:](float64[:,:], float64[:], float64[:,:])',
         fastmath=True, parallel=True)
def sff_spectra(spectra, taulist, weights):

    nsamples, nener = spectra.shape
    sfflist = np.zeros(shape=(nsamples, len(taulist)), dtype=np.complex128)
    for i in nb.prange(nsamples):
        sfflist[i] = sff_single(spectra[i], taulist, weights[i])

    return sfflist


#  Theoretical curve
def K_GOE(taulist):
    """
    GOE spectral form factor - the theoretical curve when tau is normalized
    w.r.t. the Heisenberg time

    Parameters
    ----------

    taulist: ndarray
                1D ndarray with tau parameter values
    Returns
    -------
    K_GOE: ndarray
                1D ndarray with the same len as taulist,
                contains the theoretical SFF curve.

    """

    tau_small = taulist[np.where(taulist <= 1)]
    tau_large = taulist[np.where(taulist > 1)]

    K_small = 2 * tau_small - tau_small * np.log(1 + 2 * tau_small)
    K_large = 2 - tau_large * np.log((2 * tau_large + 1) / (2 * tau_large - 1))

    return np.append(K_small, K_large)


#  A routine for extracting Thouless time:
def ext_t_thouless(taulist, sff, epsilon, smoothing):
    """
    A routine for extracting the thouless
    time, which is the time when the
    numerically obtained SFF curve falls
    onto the theoretical curve.

    Parameters
    ----------

    taulist: ndarray
                A ndarray of tau parameter
                values for which the sff
                values were numerically
                calculated
    sff: ndarray
                ndarray of the same shape
                as taulist, containing the
                sff data normalized in such
                a way that the long-time
                limit of sff equals to one.
                Pay special care to this
                condition if any kind of
                filtering has been applied.
                Can be either unconnected or
                connected sff.
    epsilon: float
                precision condition when
                determining the thouless time.

    smoothing: int
                The length of the inverval over
                which the running mean is performed.

    Returns
    -------

    taulist[minval]: float
                Thouless time value
    sff[minval]: float
                sff value at Thouless time.

    """
    sff = hlp.running_mean(sff, smoothing)
    taulist = hlp.running_mean(taulist, smoothing)

    theory = K_GOE(taulist / (2 * np.pi))
    diff = np.log10(np.abs(theory - sff) / theory)
    minval = np.argmin(diff)

    return taulist[minval], sff[minval]


class SFF_checker(object):

    """
    A class with routines and methods
    for checking sff and extracting quantities
    such as the thouless time.

    Parameters:

    taulist: ndarray
            1D ndarray of tau parameter values.
    sff: ndarray
            1D ndarray of sff values, the code
            makes sure the shape of sff
            should match that of taulist.
    sff_uncon: ndarray
            1D ndarray of unconnected part
            of sff which one has to subtract
            in order to obtain the connected
            sff
    misc_dict: dict
            A dict of misc values, should contain
            entries:
            'dims_eff' - effective dimension
            'normal_con' - normalization constant
                           for the connected part
            'normal_uncon' - normalization constant
                             for the unconnected part
            We need those for normalization.

    """

    def __init__(self, taulist, sff, sff_uncon, misc_dict,
                 *args, **kwargs):
        super(SFF_checker, self).__init__()

        if taulist.shape != sff.shape:
            raise ValueError('taulist and sff shape do not match!')
        if sff.shape != sff_uncon.shape:
            raise ValueError('sff and sff_uncon shapes do not match!')

        self.taulist = taulist
        self.sff = sff
        self.sff_uncon = sff_uncon

        keys = ['dims_eff', 'normal_con', 'normal_uncon']
        for key in keys:
            try:
                setattr(self, key, misc_dict[key])
            except KeyError:
                print('Key {} not in misc_dict. Initializing to one!'.format(
                    key))
                setattr(self, key, 1.)

    @staticmethod
    def _get_bounds(window, taulist):
        """
        A helper routine for get_thouless_time
        which properly formats array boundaries
        so that the output dimensions match after
        the running statistics operators have
        been performed.


        """
        window = int(window)
        if window % 2 == 0:  # if smoothing is even
            low = int(window / 2. - 1)
            up = - int(window / 2.)
        else:
            low = int((window - 1.) / 2.)
            up = -low

        if window == 1:
            low = 0
            up = len(taulist) + 1

        return low, up

    def get_thouless_time(self, epsilon=0.08,
                          connected=True, smoothing_window=1):
        """
        A routine for extracting thouless time from the SFF_data.
        The routine performs a calculation of the running standard
        deviation. Based on observation of our numerical data, the
        GOE regime starts when local standard deviation becomes
        noticeable. NOTE: WE MUST YET COME UP WITH AN APPROPRIATE
        JUSTIFICATION!



        Where epsilon is some desired tolerance.

        Parameters
        ----------

        epsilon: float
                Tolerance in determining the relative error treshold.
                Defaults to 1e-07.

        connected: boolean
                Whether to calculate thouless time for the connected
                or unconnected form factor.

        smoothing_window: int
                The width of the smoothing window which we sometimes
                use in order to smooth the fluctuating SFF data.
                When smoothing is applied, a running mean with the
                window of width equal to smoothing parameter is
                performed. Defaults to 1, which is equal to no
                smoothing applied.

        Returns
        -------

        tau_t: float
                Thouless time
        sff_t: float
                Sff value at thouless time
        taulist: ndarray
                1D ndarray of taulist values, possibly rescaled
                because of the smoothing procedure.
        sff_mean: ndarray
                Smoothened array of sff values
        diff: ndarray
                Array of absolute value of the difference
                between the base 10 logarithms of numerical
                values and the theory.

        """

        # two cases depending on whether we examine connected
        # or unconnected SFF
        if not connected:
            sff = self.sff / self.dims_eff
        else:
            sff = (self.sff - (self.normal_con / self.normal_uncon) *
                   self.sff_uncon) * (1. / self.dims_eff)

        # ------------------------------------------------------------
        # SMOOTHING CALCULATION
        # ------------------------------------------------------------
        taulist = self.taulist.copy()
        low, up = self._get_bounds(smoothing_window, taulist)

        # diff = sff  # (np.abs(sff - theory) / np.abs(theory))
        # diff = np.abs(sff - theory)
        # perform smoothing of the raw data (minus the theoretical
        # value) first

        sff = pd.Series(sff).rolling(window=smoothing_window, center=True)
        sff_mean = sff.mean().values

        # filter out nan values
        sff_mean = sff_mean[~np.isnan(sff_mean)][::-1]

        taulist = taulist[low:up]

        taulist *= 1. / (2 * np.pi)
        theory = K_GOE(taulist)[::-1]

        diff = np.abs(np.log10(sff_mean) - np.log10(theory))
        tau_arg = np.argwhere(diff > epsilon)[0][0]
        # tau_arg = np.argwhere(diff_std > epsilon)[0][0]

        tau_t = taulist[::-1][tau_arg]
        sff_t = sff_mean[tau_arg]

        return tau_t, sff_t, taulist, sff_mean[::-1], diff[::-1]
