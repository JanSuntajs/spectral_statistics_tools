"""
A module with functions used for
performing different types of spectral
unfolding whihc is needed in various
examples of spectral statistics
calculations where the effects of
nonuniform density of states need
to be diminished.


"""

import numpy as np
from utils import tester_methods as _tst
import warnings as _warnings


class Unfold_mixin(object):
    """
    A "mixin" class which only contains
    methods but deliberately lacks an
    __init__ method. Usage of mixin classes
    is intended for easier breaking of classes
    into smaller units each containing
    appropriate methods.




    """

    def _unfold_function(self, n=3, merge_spectra=False):
        """
        A function that performs spectral unfolding so that the mean level
        spacing after the procedure equals one. The routine is needed in
        various spectral statistics calculations, such as calculations of
        the spectral form factor, calculation of the number level variance
        or calculations of the distribution of the adjacent energy level
        spacings. By unfolding the spectrum, we get rid of the scaling
        effects in the spectra and can thus compare the energy level
        statistics of systems with different sizes/Hilbert space
        dimensions and bandwidths.

        A brief explanation of the procedure:
        a cumulative distribution of energies is fitted by the polynomial of
        degree n -> p_n(energy). The values of the polynomial at our provided
        energies are then stored. By construction, the mean level
        spacing/energy density of the unfolded spectrum now equals one.

        Parameters
        ----------
        spectra: ndarray
                    A 2D array of energy values for an ensemble of spectra.
                    The code also works if only a single spectrum is
                    present.
        n: int
                    The degree of the fitting polynomial. The procedure
                    works the best if low-degree polynomials are used.
        spectral_width: Tuple[float, float]
                    A tuple specifying the percentage
                    of states in the spectrum that is
                    considered in the calculation. By
                    default, 50% of states near the
                    centre of the spectrum are used.
        merge_spectra: boolean
                    if True: all spectra are merged and sorted, then the
                    unfolding procedure (e.g. finding the optimum polynomial
                    curve) is performed on the merged spectrum. The curve
                    is then applied to individual spectra.
                    if False: analysis is performed on each spectrum
                    separately.

        Returns
        -------
        unfolded: ndarray
                    A 2D array of unfolded spectra which were possibly resized
                    as well according to the input provided in the
                    spectral_width parameter.


        """

        def fitter_fun(x, y, n, return_vals=False):

            with _warnings.catch_warnings():
                _warnings.filterwarnings("error")
                z = np.polyfit(x, y, n)
                p = np.poly1d(z)

            if return_vals:
                return p(x)
            else:
                return p

        # _tst._check_spectral_width(spectral_width)
        # spectra = _tst._check_spectral_dimensions(spectra, 2)
        spectra, nsamples, nener = _tst._resize_spectra(self._spectrum0,
                                                        self.spectral_width)

        cum_dist = np.arange(1, nener + 1, 1)

        if merge_spectra:
            # All spectra are first merged, then sorted - here, merging means
            # reshaping a 2D array into 1D shape. Unfolding procedure is then
            # performed on the merged spectrum in order to obtain the fitting
            # polynomial function. The latter is then applied to individual
            # spectra for different disorder realizations.

            fitlist = np.sort(spectra.reshape(-1,))
            cum_dist = np.arange(1., fitlist.size + 1., 1.) / nsamples
            p = fitter_fun(fitlist, cum_dist, n)

            unfolded = np.apply_along_axis(p, 1, spectra)
        else:
            # If no merging of spectra is performed, the unfolding procedure
            # is performed on each individual spectrum.

            unfolded = np.apply_along_axis(fitter_fun, 1, spectra, cum_dist,
                                           n, True)

        return unfolded

    @staticmethod
    def _correct_slope(spectra):
        """
        A helper routine for the unfolding procedure that makes
        sure the energies in the unfolded spectra are always
        in the ascending order. This means that the tails of the
        spectra can be cut, if needed.
        NOTE 1: using this procedure
        only makes sense on unfolded spectra!
        NOTE 2: using this procedure is only appropriate if the
        oscillations due to the polynomial fitting procedure
        during the unfolding are near the edges of the spectra.
        A low-degree polynomial must be used in fitting to ensure
        this.


        Parameters
        ----------
        spectra: ndarray
                    A 2D array of unfolded energy spectra
        Returns
        -------
        spectra: ndarray
                    A 2D array of unfolded energy spectra with
                    corrected slope

        """
        # Find where the slope is negative
        nener = spectra.shape[1]
        min_idx = np.max(np.argmax(np.diff(spectra, axis=1) > 0, axis=1))

        max_idx = np.max(
            np.argmax(np.diff(spectra, axis=1)[:, ::-1] > 0, axis=1))

        max_idx = nener - max_idx
        spectra = spectra[:, min_idx:max_idx]

        return spectra

    def spectral_unfolding(self, n=3, merge=False, correct_slope=False):
        """
        A function that performs spectral unfolding - a procedure that
        sets the average energy level density to the value of one and
        allows for comparison of different systems' spectra, for
        instance, for different system sizes or bandwidths. In this
        implemmentation, the function is basically a wrapper for the
        utils.unfolding module routines.

        Parameters
        ----------
        self

        n: int (optional)
                    The fitting polynomial degree, defaults to n=3. In order
                    to avoid possible issues with ill-conditioned polynomials,
                    n should be rather small. Some testing is needed to find
                    the optimum value, however, the default value works quite
                    well in most cases.
        merge: boolean (optional)
                    If True, all spectra are first merged - reshaped from a
                    2D into a 1D array - then sorted and the unfolding
                    procedure is then performed on the merged spectrum
                    in order to find the optimum fitting polynomial.
                    Once the polynomial function is found, it is applied
                    back to individual spectra. If False, the optimum
                    fitting function is calculated for each polynomial
                    individually. The default is set to False, as we have
                    found that merge fitting produces incorrect results
                    for the number level variance calculations, but we
                    nevertheless keep this as an additional option.
        correct_slope: boolean (optional)
                    Whether to correct the unfolded spectra for possible
                    unphysical effects or not. Due to the unfolding
                    procedure, some portions of the unfolded spectrum
                    may be such that the energies appear in a descending
                    order. If correct_slope = True, a procedure is called
                    which cuts of the spectral tails in which this
                    generally occurs due to oscillations in the fitting
                    polynomial. This scheme only works if the oscillations
                    appear near the spectral tails. For low-degree fitting
                    polynomials, this is generally the case.

        Returns
        -------

        Updates self.unfolded with the values of unfold_dict.

        """

        # check if unfolding parameters have changed since the
        # last time

        print(('Performing spectral unfolding '
               'with the following settings: \n'
               'n: {}, merge: {}, correct_slope: {} \n'
               'spectral_width: {}'
               ).format(n, merge, correct_slope, self.spectral_width))

        unfold_dict = {}

        try:
            unfolded = self._unfold_function(n, merge)
        except (RuntimeWarning, np.RankWarning) as e:
            print('{}: {}'.format(type(e).__name__, e))
            msg = ("Polynomial degree n={} is to high "
                   "so the fitting polynomial is "
                   "ill-conditioned! Reverting to "
                   "n=1.").format(n)
            _warnings.warn(msg)
            n = 1
            unfolded = self._unfold_function(n, merge)

        discarded = unfolded.shape[1]

        if correct_slope:
            unfolded = self._correct_slope(unfolded)

        discarded -= unfolded.shape[1]

        self._spectrum = unfolded

        unfold_dict['n'] = n
        unfold_dict['merged'] = merge
        unfold_dict['correct_slope'] = correct_slope
        unfold_dict['spectral_width'] = self.spectral_width
        #  how many values were discarded during
        #  procedure due to slope correction
        unfold_dict['discarded'] = discarded

        # If unfolding of the spectrum is performed, make
        # sure that other quantities which are calculated
        # on the basis of the unfolded data are reset.
        self._unfolding_performed = True
        self._resizing_performed = True
        self._misc_calculated = False
        self._filtering_performed = False

        self._unfold_dict = unfold_dict
        self._misc_dict = None
        self._filter_dict = None

    def spectral_resizing(self):
        """
        A function that only resizes the
        spectra so that the desired
        percentage is contained.

        Resizing is performed on the original
        spectrum.
        """
        spectrum, nsamples, nener = _tst._resize_spectra(self._spectrum0,
                                                         self.spectral_width)

        self._spectrum = spectrum

        self._resizing_performed = True
        self._misc_calculated = False
        self._filtering_performed = False

        self._unfold_dict = None
        self._misc_dict = None
        self._filter_dict = None
