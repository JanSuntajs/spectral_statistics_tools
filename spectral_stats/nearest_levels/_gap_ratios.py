"""
A module with functions used in analyzing
ratios of consecutive level spacings.

"""

import numpy as np
import numba as nb

from ..utils import tester_methods as _tst


# use numba for speed
@nb.njit('float64[:](float64[:])', fastmath=True, parallel=True)
def _gaps_numba(gaps):
    ratios = np.ones(shape=len(gaps) - 1, dtype=np.float64)
    for i in nb.prange(len(gaps[:-1])):
        pair = np.array([gaps[i], gaps[i + 1]], dtype=np.float64)
        ratios[i] = np.min(pair) / np.max(pair)

    return ratios


def _calc_gaps(spectrum, spectral_width=(0.25, 0.75)):
    """
    A function that calculates the ratios of the
    adjacent energy gaps in the spectrum.

    Parameters
    ----------
    spectrum: ndarray
                1D array of floats
                containing the energy values
                of the spectrum sorted in
                ascending order. Can be any
                iterable that can be converted
                to a 1d numpy ndarray

    spectral_width: Tuple[float, float]
                A tuple specifying the percentage
                of states in the spectrum that is
                considered in the calculation. By
                default, 50% of states near the
                centre of the spectrum are used.


    Returns
    -------
    ratios: ndarray
                1D array containing values of ratios
                of the adjacent level spacings. Length
                of the array should equal
                len(spectrum) - 2.
    avg_ratio: float
                Average ratio of the whole spectrum.

    Examples
    --------
    >>> calc_gaps([1,2,3,4,5], (0., 1.))
    (array([1., 1., 1.]), 1.0)
    >>> calc_gaps([1.,1.,2.], (0.,1.))
    Traceback (most recent call last):
        ...
    ValueError: Degeneracies are present in the spectrum!
    >>> calc_gaps([1,2,3,4,5],(-0.1,1.1))
    Traceback (most recent call last):
        ...
    ValueError: spectral_width should have values between 0 and 1.
    >>> calc_gaps([1,2,3,4,5],(0.3,0.2))
    Traceback (most recent call last):
        ...
    ValueError: Upper spectral limit should be greater than the lower one.
    >>> calc_gaps([],(0.3,0.2))
    Traceback (most recent call last):
        ...
    _EmptyArrayError: Spectrum array is empty!
    >>> calc_gaps([[1.,2.,3.]], (0.,1.))
    Traceback (most recent call last):
        ...
    ValueError: Spectrum should be a 1-dimensional array!

    """

    # make sure the input data are of the array type
    spectrum = _tst._check_spectral_dimensions(spectrum, 1)
    _tst._check_spectral_width(spectral_width)

    low, up = spectral_width
    num_energies = len(spectrum)

    spectrum = spectrum[int(low * num_energies):int(up * num_energies)]
    gaps = np.diff(spectrum)

    if any(gaps == 0.):
        raise ValueError('Degeneracies are present in the spectrum!')

    # ratios = np.ones(len(gaps) - 1, dtype=np.float)

    # for i, gap in enumerate(gaps[:-1]):
    #     pair = (gaps[i], gaps[i + 1])
    #     ratios[i] = min(pair) / max(pair)
    ratios = _gaps_numba(gaps)

    avg_ratio = np.mean(ratios)
    return ratios, avg_ratio


class Gap_mixin(object):
    # The actual functions for calculating the gap ratios

    def gap_avg(self, return_distribution=False):
        """
        A function that calculates the average gap ratio
        for a given ensemble of spectra, which were, for
        instance, obtained for different realizations of
        disorder in an MBL problem.

        Parameters
        ----------

        spectra: ndarray
                    2D array of energy spectra.
        spectral_width: Tuple[float, float]
                    A tuple specifying the percentage
                    of states in the spectrum that is
                    considered in the calculation. By
                    default, 50% of states near the
                    centre of the spectrum are used.

        return_distribution: boolean, optional
                    Whether to return the distribution
                    of average r values for different
                    sample realizations.
        Returns
        -------
        gap_mean: float
                    Mean value of the average gap ratio
                    where mean is obtained over different
                    spectra.
        gap_dev:  float
                    Statistical deviation of the calculated
                    gap mean.

        ratiolist: ndarray
                    A ndarray of type float returning the mean
                    ratio for each individual sample. Only returned
                    if return_distribution == True.
        Examples
        --------
        >>> gap_avg([1,2,3,4,5], (0., 1.))
        (1.0, 0.0)
        >>> gap_avg([[1,2,3,4,5]], (0., 1.))
        (1.0, 0.0)
        >>> gap_avg([[]], (0., 1.))
        Traceback (most recent call last):
            ...
        _EmptyArrayError: Spectrum array is empty!

        """

        # spectra = _tst._check_spectral_dimensions(self._spectrum, 2)
        # _tst._check_spectral_width(self.spectral_width)

        ratiolist = np.array([_calc_gaps(spectrum, self.spectral_width)[1] for
                              spectrum in self._spectrum0])
        gap_mean = np.mean(ratiolist)
        gap_dev = np.std(ratiolist)
        if return_distribution:
            return gap_mean, gap_dev, ratiolist
        else:
            return gap_mean, gap_dev

    def gap_dist(self,
                 **kwargs):
        """
        A function returning r values for all the disorder samples
        which can then be statistically analysed in terms of their
        distribution.


        Returns
        -------
        ratiolist: 1D ndarray
                  A flattened array of r values for all the disorder
                  samples

        """

        ratiolist = np.array([_calc_gaps(spectrum, self.spectral_width)[0] for
                              spectrum in self._spectrum0])

        ratiolist = ratiolist.flatten()

        return ratiolist
