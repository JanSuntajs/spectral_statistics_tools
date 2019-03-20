
from collections import defaultdict

# from nearest_levels import _gap_ratios as _gr
from nearest_levels._gap_ratios import Gap_mixin
from utils._unfolding import Unfold_mixin
from utils._misc import Misc_mixin
from sff._sff import Sff_mixin
from utils import tester_methods as _tst
import numpy as _np


def _makehash():
    """
    A function that enables creation of nested dictionaries
    of any desired depth.
    """
    return defaultdict(_makehash)


class Spectra(Sff_mixin, Misc_mixin, Unfold_mixin, Gap_mixin):
    """
    The spectrum object contains an ensemble of energy spectra on which
    various spectral statistics can be performed.

    Parameters
    ----------

    spectrum: ndarray
                A 2D ndarray containing energy spectra for which the
                spectral statistics are to be calculated.

    Attributes
    ----------

    _spectrum: ndarray
                A 2D ndarray that stores the original ensemble of spectra
                before any operations have been performed on them.
    spectrum: ndarray
                A 2D ndarray that stores the "working version" of the
                spectrum, on which unfolding or spectral resizing (or both)
                have been performed.
    nener: int
                The number of energies (the Hilbert space dimension) in the
                original spectra. _spectrum.shape[1]
    nsamples: int
                The number of individual spectra that are stored in the
                spectrum array. _spectrum.shape[0]

    spectral_width: Tuple[float, float]
                The ratio of states to consider in spectral analysis
                calculations. low = spectral_width[0] is the lower boundary
                of the spectrum, while up = spectral_width[1] is the upper
                boundary. The values should be between 0 and 1 and sorted in
                ascending order. This is used since we often wish to
                consider only a certain portion of states near, say, the
                centre of the spectrum. We do this as follows:
                spectrum_resized = self.spectrum[:,
                                                  int(low * self.nener):
                                                  int(up * self.nener)]
    _unfolded_performed: boolean
                Tells whether unfolding was performed or not. Initializes
                to False.

    _misc_calculated: boolean
                Tells whether misc quantities were already calculated for
                a given spectrum and spectral width
    _filtering_performed: boolean
                Tells whether spectral filtering has already been performed
                for a given spectrum, spectral width and misc quantities
    unfolded: {None, dict}
                A dict containing the unfolded spectrum and data
                about the unfolding procedure. If None, the
                unfolding procedure has not been called yet.
                The format of the dict with corresponding
                keys and values:
                'spectrum': 2D ndarray with the unfolded values
                'spectral_width': tuple, spectral_width for which the
                                  unfolded spectrum was calculated
                'n': int, the fitting polynomial degree
                'merge': boolean, True, False. Whether spectra
                         were merged or not
                'correct_slope': boolean, True, False. Whether
                                  slope was corrected or not
                'discarded': int. How many values were lost
                            Due to the slope correction
                            procedure.



    """

    def __init__(self, spectrum, *args, **kwargs):
        super(Spectra, self).__init__()

        self._unfolding_performed = False
        self._misc_calculated = False
        self._resizing_performed = False
        self._filtering_performed = False
        self._unfold_dict = None
        self._misc_dict = None
        self._filt_dict = None
        self.spectral_width = (0., 1.)
        self._spectrum0 = spectrum
        # self._spectrum = _np.zeros_like(spectrum,
        #                                 dtype=_np.float64)

    @property
    def spectral_width(self):

        return self._spectral_width

    @spectral_width.setter
    def spectral_width(self, spectral_width):

        _tst._check_spectral_width(spectral_width)

        self._spectral_width = spectral_width

    @property
    def _spectrum0(self):
        """
        __spectrum0 is the initial
        spectrum, which we do not
        touch!
        """
        return self.__spectrum0

    @_spectrum0.setter
    def _spectrum0(self, spectrum):
        """
        Check if the dimensions of the input data
        are ok.
        """
        spectrum = _tst._check_spectral_dimensions(spectrum, 2)

        self.__spectrum0 = spectrum

    @property
    def spectrum(self):

        if (not self._unfolding_performed):
            spectra, nsamples, nener = _tst._resize_spectra(
                self._spectrum0,
                self.spectral_width)

            self._resizing_performed = True
            return spectra

        else:
            return self._spectrum

    @property
    def _nener(self):
        """
        Number of energies im the
        original spectrum.
        """
        return self.__spectrum0.shape[1]

    @property
    def _nsamples(self):
        """
        Number of samples im the
        original spectrum.
        """
        return self.__spectrum0.shape[0]

    @property
    def nener(self):
        """
        Number of energies im the
        "working" spectrum.
        """
        return self.spectrum.shape[1]

    @property
    def nsamples(self):
        """
        Number of samples im the
        "working" spectrum.
        """
        return self.spectrum.shape[0]

    @property
    def unfolded(self):
        if not self._unfolding_performed:
            print(('Warning! The unfolding procedure '
                   'has not yet been called! The return '
                   'in this case defaults to None. '))
        return self._unfolded

    @property
    def unfold_dict(self):

        return self._unfold_dict

    @property
    def misc_dict(self):

        return self._misc_dict

    @property
    def filt_dict(self):

        return self._filt_dict

    def hist(self, bins):
        """
        A function that returns the
        histogram of an averaged working
        spectrum.

        Parameters
        ----------
        bins: int
                    Number of histogram bins
        Returns:
        --------
        hist: ndarray
                    Histogram values.
        edges: ndarray
                    Histogram bin edges.

        Note:
        -----
        For plotting the return, use edges[:-1], hist
        """
        ave = _np.mean(self.spectrum, axis=0)
        # bounds = _np.diff((_np.min(ave), _np.max(ave)))
        hist, edges = _np.histogram(ave, bins=bins, density=True)

        return hist, edges

    #  ---------------------------------------------------------
    #   SFF calculations
    #
    #  ---------------------------------------------------------
