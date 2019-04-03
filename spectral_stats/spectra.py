"""
This module implements the Spectra class which implements
methods for calculations of various spectral statistical
observables which one typically encounters in studies of
the energy spectra of ergodic or many-body localized
systems. The expected use case is an analysis of an
ensemble of energy spectra of many-body Hamiltonians, which
were obtained by means of some external method/program,
using for instance the approaches of full or partial
diagonalization.

The Spectra class contains methods which allow for
calculations of:

- mean ratio of the adjacent energy level spacings
- spectral unfolding
- miscellaneous quantities, such as:
      - Hamiltonian's trace
      - trace of the squared Hamiltonia
      - Hamiltonian's mean energy
      - energy spectra's gamma values
- spectral form factor
- spectral filtering used in spectral-form factor
  calculations

TO DO: COMPLETE THE DOCUMENTATION; AT THIS STAGE,
IT IS INCONSISTENT AND INCOMPLETE

"""

from collections import defaultdict

import numpy as _np
import os
import time

from .nearest_levels._gap_ratios import Gap_mixin
from .utils._unfolding import Unfold_mixin
from .utils._misc import Misc_mixin
from .sff._sff import Sff_mixin
from .level_variance._lvl_var import Lvl_mixin
from .utils import tester_methods as _tst
from dataIO.hdf5saver import hdf5save


_text_description = """
This hdf5 file stores various spectral statistics calculated on an \n
ensemble of energy-spectra of many-body hamiltonians. In the following \n
a brief description of file's contents is given. \n

The hdf5 file itself contains multiple datasets, which are accessed in a \n
manner similar to accesing a python dictionary. We recommend using the user
defined hdf5load() function from the dataIO.hdf5saver module, which is \n
distributed alongside the spectral_stats package. The datasets are: \n
- spectrum: a 2D ndarray, stores the modified initial ensemble of the \n
energy spectra, on which the unfolding and spectral resizing have possibly \n
been performed. \n
- mean_ener, sq_ham_tr, ham_tr_sq, gamma: 1D ndarrays of the same shape,\n
their length equals the number of spectra in the ensemble. They store\n
the value of the mean energy, square of the Hamiltonian's trace, trace\n
of the squared Hamiltonian and the gamma value of the spectra. Depending\n
on the choice made during the calculation, the values can be either given\n
as an average over all spectra or can be calculated for each spectrum\n
individually. In either case, the values are returned as arrays, not scalars.\n
See also the explanation of the 'individual_misc' entry in the 'misc'
dataset.\n
- filter: 2D ndarray of the same shape as the spectrum array. In the spectral\n
form factor calculations, spectral filtering can be applied in order to\n
mitigate the finite-size scaling effects. The filter array stores the filter\n
that has been applied to the energy spectrum in the spectral form factor\n
calculations. Currently, we only have the identity filter (e.g. no filtering)\n
and the Gaussian filter implemented. In the latter case, the width of the\n
Gaussian is determined as the user selected parameter eta * gamma, where\n
gamma is the quantity which we have defined above.\n
- taulist, sff, sff_uncon: 1D ndarrays, all of the same shape as\n
taulist, which store the data about the tau parameter values, the sff\n
which includes the so-called unconnected part, and the values of the\n
unconnected sff part which one needs to subtract in order to obtain\n
the connected spectral form factor.\n
- lvals: 1D ndarray with energy values on which the number level\n
variance is calculated. If level variance was not calculated, this\n
is an empty ndarray.\n
- lvl_var: 2D ndarray of shape (3, len(lvals)) or (6, len(lvals)).\n
In the first case, the average number of levels in an energy interval\n
is returned, then the mean number level variance and the variance\n
of the mean level variance. In the second case, also the statistical\n
errors due to averaging over different random samples are returned.\n

Apart from these main quantities, there are also the 'misc' and\n
'metadata' datasets. The 'misc' dataset contains the data related
to the technical details of the calculations which led towards\n
the results described above. The contents of a misc dataset are:\n
-n: int, the degree of the fitting polynomial used during the \n
unfolding procedure\n
-merged_unfolding: boolean. If True, all the spectra in the\n
ensemble were merged before the fitting polynomial for the\n
unfolding was found.\n
-correct_slope: boolean. If True, slope correction procedure\n
was used during the unfolding procedure.\n
-discarded_unfolding: int. How many eigenvalues have been\n
discarded due to the slope correction procedure during unfolding.\n
-spectral_width: Tuple[float, float]. The percentage of states\n
from the original spectrum that was used in the working/modified\n
spectrum.\n
-individual_misc: boolean. If True, the quantities in the 'misc'\n
dataset were calculated for each spectrum individually. If False,\n
they were calculated as an average over all the ensemble's spectra.\n
-filter_type: string. Which type of filtering was used during the\n
sff calculations.\n
eta: float. Only present if Gaussian filtering was used. A param.\n
which determines the width of the Gaussian filter w.r.t. the width\n
of the spectrum, which is given by Gamma defined above.\n
-nener, nsamples - int, int. the Hilbert space dimension and the
number of samples in the working/modified spectrum.\n
-nener0, nsamples0 - the same as in the previous entry, only for\n
the original spectrum on which no operations have been performed.\n
-dims_eff: float. If spectral filtering was used, the normalization\n
constant for the spectral form factor is not simply equal to the\n
Hilbert space dimension. Instead, we use the "effective Hilbert\n
space dimension".\n
-normal_con, normal_uncon: float, float. The normalization\n
constants for connected and unconnected spectral form factor.\n
Can become relevant if filtering is used and one wants to subtract\n
the unconnected contribution to the spectral form factor.
"""


metadata = {

    'OS': os.name,
    'User': 'Jan Suntajs',
    'email': 'Jan.Suntajs@ijs.si',
    'institute': 'IJS F1',
    'Date': time.time(),
    'dict_format': 'json',
    'description': _text_description

}


def _makehash():
    """
    A function that enables creation of nested dictionaries
    of any desired depth.
    """
    return defaultdict(_makehash)


class Spectra(Lvl_mixin, Sff_mixin, Misc_mixin, Unfold_mixin, Gap_mixin):
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

    _spectrum0: ndarray
                A 2D ndarray that stores the original ensemble of spectra
                before any operations have been performed on them.
    spectrum: ndarray
                A 2D ndarray that stores the "working version" of the
                spectrum, on which unfolding or spectral resizing (or both)
                have been performed.
    _nener: int
                The number of energies (the Hilbert space dimension) in the
                original spectra. _spectrum.shape[1]
    _nsamples: int
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

    unfold_dict: {None, dict}
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

        # control variables
        self._unfolding_performed = False
        self._misc_calculated = False
        self._resizing_performed = False
        self._filtering_performed = False
        self._sff_calculated = False
        self._lvl_var_calculated = False

        # unfolding, misc, filtering
        self._unfold_dict = {}
        self._misc_dict = {}
        self._filt_dict = {}

        # sff
        self.taulist = _np.ndarray([])
        self.sff = _np.ndarray([])
        self.sff_uncon = _np.ndarray([])

        # lvl_variance
        self.lvl_var = _np.ndarray([])
        self.lvals = _np.ndarray([])

        # defaults
        self.spectral_width = (0., 1.)
        self._spectrum0 = spectrum

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
    def misc0_dict(self):
        """
        Return dict with miscellanea
        quantities for the original spectrum.

        Average values of the quantities for
        the whole ensemble are returned in a
        flattened  form.
        """

        return self._ham_misc(self._spectrum0, False, True)

    @property
    def spectrum(self):

        if (not self._resizing_performed):
            spectra, nsamples, nener = _tst._resize_spectra(
                self._spectrum0,
                self.spectral_width)

            self._resizing_performed = True
            self._spectrum = spectra

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

    def save(self, filename, metadata=metadata, system_info={},
             *args, **kwargs):
        """
        A function for saving data in the HDF5
        format using the user defined hdf5saver
        module

        We recommend loading the file back using the
        dataIO.hdf5saver.hdf5load function which returns
        a dict of values.
        """

        datasets = {

            'spectrum': self.spectrum,
            'mean_ener': self.misc_dict['mean_ener'],
            'sq_ham_tr': self.misc_dict['sq_ham_tr'],
            'ham_tr_sq': self.misc_dict['ham_tr_sq'],
            'gamma': self.misc_dict['gamma'],
            'filter': self.filt_dict['filter'],
            'sff': self.sff,
            'sff_uncon': self.sff_uncon,
            'taulist': self.taulist,
            'lvals': self.lvals,
            'lvl_var': self.lvl_var
        }

        filt_exclude = ['filter', 'dims']
        filt_dict = dict((key, self.filt_dict[key])
                         for key in self.filt_dict if key not in filt_exclude)
        misc_exclude = ['mean_ener', 'sq_ham_tr', 'ham_tr_sq', 'gamma']

        misc_dict = dict((key, self.misc_dict[key])
                         for key in self.misc_dict if key not in misc_exclude)
        misc0 = self.misc0_dict.copy()
        misc0_keys = [key for key in misc0]
        for key in misc0_keys:
            misc0[key + '0'] = misc0.pop(key)

        attrs = self.unfold_dict.copy()
        for dict_ in (misc_dict, filt_dict, misc0):
            attrs.update(dict_)

        attrs.update({'nener': self.nener, 'nsamples': self.nsamples,
                      'nener0': self._nener, 'nsamples0': self._nsamples})

        print(attrs)
        hdf5save(filename, datasets, attrs, metadata, system_info)
