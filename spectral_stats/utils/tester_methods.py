"""
A module that implements commonly used
methods for testing conditions such as
the ones on the input arguments' format
and shape

"""

import numpy as _np


class _EmptyArrayError(Exception):
    """ Throw this when a list of energy values
    is empty"""
    pass


def _check_spectral_dimensions(values, dim):
    """
    A helper routine that checks if an input
    array is of proper shape and of nonzero
    size. If those tests are passed, it returns
    the input values as a numpy array. If
    dim == 2 and values.ndim == 1, handle this
    case separately so that calc_gaps() function
    does not return an error but calculates
    the average gap ratio on a single spectrum
    instead.

    Parameters
    ----------

    values: ndarray
                dim-dimensional array of data
    dim: int
                dimensionality of the problem
    Returns
    -------

    values: ndarray
                input data formatted as a numpy
                array.

    """

    values = _np.asarray(values, dtype=_np.float)

    if (dim == 2 and values.ndim == 1):
        values = values.reshape(1, -1)

    if values.ndim != dim:
        raise ValueError(
            'Spectrum should be a {}-dimensional array!'.format(dim))
    if values.size == 0:
        raise _EmptyArrayError('Spectrum array is empty!')

    return values


def _check_spectral_width(spectral_width):
    """
    A function that performs test on the
    spectral width tuple to see if the
    inputs are of proper format.
    """

    if any([(x < 0 or x > 1) for x in spectral_width]):
        raise ValueError('spectral_width should have values between 0 and 1.')

    if any(_np.diff(spectral_width) < 0):
        raise ValueError(
            'Upper spectral limit should be greater than the lower one.')


def _resize_spectra(spectra, spectral_width):
    """
    A function that extracts a portion of energy
    states between spectral_width[0]*100 and
    spectral_width[1]*100 percent of the spectrum

    NOTE: use this after _check_spectral_dimension()
    and _check_spectral_width have been called.

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
    Returns
    -------
    spectra: ndarray
                resized spectra
    nsamples: int
                number of samples in spectra
    nener: int
                number of energies in a single
                spectrum after resizing
    """
    _check_spectral_width(spectral_width)
    spectra = _check_spectral_dimensions(spectra, 2)
    
    low, up = spectral_width
    nsamples, nener = spectra.shape
    spectra = spectra[:, int(low * nener):int(up * nener)]
    nsamples, nener = spectra.shape
    return spectra, nsamples, nener
