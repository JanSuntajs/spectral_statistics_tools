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
