"""
A module with routines for calculating
miscellaneous spectral data and for
creating filtering functions on the basis
of energy values in the spectrum.

The mixin classes in this package are defined
and used in such a way that the main working
class is a subclass of multiple mixin classes
from which it inherits the relevant methods.

Some general requirements for a subclass inheriting
from a mixin class in this package:

A subclass needs to the following attributes:

    _spectrum: ndarray
                A 2D ndarray that stores the original ensemble of spectra
                before any operations have been performed on them.

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

"""
from future.utils import iteritems

import numpy as np

from ._unfolding import Unfold_mixin
from . import filter_functions as _fif


class Misc_mixin(Unfold_mixin):

    @staticmethod
    def _ham_misc(spectra, individual=False, flatten=False):
        """
        A helper routine for calculating:

        - Mean energy (trace of the Hamiltonian)
        - Square of the mean energy (square of the Hamiltonian's trace)
        - Trace of the Hamiltonian squared
        - the effective width of the Hamiltonian spectrum which we denote
        Gamma:
          Gamma ** 2 = tr(ham**2/dims) - (tr(ham/dims))**2
        NOTE: the reason we make the above distinctions lies in the need for
        averaging the values over the ensemble of similar Hamiltonians.

        Parameters
        ----------

        spectra: ndarray
                    A 2D ndarray of energy spectra
        individual: boolean
                    Whether to perform calculation on each spectrum
                    individually or to perform it on all spectra at
                    once and return the mean value
        flatten: boolean
                    if individual is true, return a scalar value
                    instead of a vector of quantities for each
                    individual spectrum
        Returns
        -------
        misc_dict: dict
                    A dictionary of the above quantities.
        """

        dict_keys = ['mean_ener', 'sq_ham_tr', 'ham_tr_sq', 'gamma']
        misc_dict = {}

        mean_ener = np.mean(spectra, axis=1)
        # Square of the Hamiltonian's trace: tr(ham / dims) ** 2
        # we use this definition since we are ultimately interested in the
        # average value of gamma for the whole ensemble of spectra. Hence
        # this definition enables us to first calculate gamma for each spectrum
        # separately and then average those values.
        sq_ham_tr = np.mean(spectra, axis=1) ** 2
        # Trace of the squared Hamiltonian: tr(ham ** 2 / dims)
        ham_tr_sq = np.mean(spectra**2, axis=1)
        # Gamma
        gamma = np.sqrt(ham_tr_sq - sq_ham_tr)

        quantities = [mean_ener, sq_ham_tr, ham_tr_sq, gamma]

        if not individual:

            if flatten:
                quantities = [np.mean(quantity)
                              for quantity in quantities]
            else:
                quantities = [np.full(spectra.shape[0], np.mean(quantity), )
                              for quantity in quantities]

        for i, key in enumerate(dict_keys):
            misc_dict[key] = quantities[i]

        return misc_dict

    def get_ham_misc(self, individual=False):
        """
        A function that returns various quantities of interest for the energy
        spectra under investigation:

        - Mean energy (trace of the Hamiltonian)
        - Square of the mean energy (square of the Hamiltonian's trace)
        - Trace of the Hamiltonian squared
        - the effective width of the Hamiltonian spectrum which we denote
        Gamma:
          Gamma ** 2 = tr(ham**2/dims) - (tr(ham/dims))**2
        NOTE: the reason we make the above distinctions lies in the need for
        averaging the values over the ensemble of similar Hamiltonians.

        Parameters
        ----------
        self: spectra
                    Instance of the spectra class object or any other similar
                    class which also implements the _spectrum and spectral
                    width attributes, the first one being a 2D ndarray of
                    energy spectra and the second one begin a Tuple of two
                    floats.
        individual: boolean
                    If True, the above mentioned quantities are calculated for
                    each spectrum individually and an array of values is
                    returned. If False, the mean value is returned.

        Returns
        -------
            mean_ener: ndarray
                    1D array with nsamples values of mean energies. If
                    individual is set to true, those values are in
                    general all different, in the opposite case they
                    are all equal to the mean taken over all the spectra.
                    The same holds for all other returned quantities
            sq_ham_tr: ndarray
                    1D array of the same shape as mean_ener. Returns the value
                    of the squared mean of the Hamiltonian's trace
            ham_tr_sq: ndarray
                    1D array of the same shape as mean_ener. Returns the value
                    of the trace of the squared Hamiltonian.
            gamma: ndarray
                    1D array of the same shape as mean_ener, returns the Gamma
                    values where Gamma was defined above.

        Examples
        --------
        >>> get_ham_misc([1,1,1,1], False, (0., 1.))
        ([1.], [1.], [1.], [0.])

        """

        misc_dict = self._ham_misc(self.spectrum,
                                   individual)

        misc_dict['individual_misc'] = individual

        # if self._unfolding_performed:
        #     # add extra data for the unfolded spectrum
        #     # n - the degree of the fitting polynomial
        #     # merged - if the unfolded spectra were

        #     # previously merged (check the unfolding routines)
        #     # correct_slope - if slope correction was used
        #     additional_keys = ['n', 'merged', 'correct_slope']
        #     for key in additional_keys:
        #         misc_dict[key] = self._unfold_dict[key]

        self._misc_dict = misc_dict
        self._toggle_states(3)

    @staticmethod
    def _spectral_filtering(spectra, filter_func, *args, **kwargs):
        """
        A function that returns a filter and appropriate normalization
        constants used in filtering data before calculating the
        spectral form factor function.

        The spectral filter is a function of the form
            g(E_n), where E_n are the energies in the spectrum. An example
            of filter usage is the following calculation of the spectral form
            factor (SFF):

            K(t) = < |\sum_n g(E_n)\exp(-iE_n t)|**2>

            Parameters
            ----------
            spectra: ndarray
                        A 2D ndarray of energy spectra
            filter_func: function
                        A function that accepts positional and keyword
                        arguments *args and **kwargs, respectively, and
                        returns a 2D ndarray of filtering values for
                        each single spectrum among spectra.
            Returns
            -------
            filter: ndarray
                        A 2D ndarray, return of the filter_func function.
            dims: int
                        Hilbert space dimension of the input spectra before
                        filtering has been applied - spectra.shape[1]
            dims_eff: float
                        Effective dimensions of the Hilbert space after
                        filtering has been applied. Obtained by letting
                        t -> infty in the above K(t) definition and noting
                        that the square of the modulus can be written as a
                        double sum in which only the diagonal terms
                        contribute in the t -> infty limit. We then have:
                        dims_eff = <\sum_n (|g(E_n)|**2) >
            normal_con: float
                        Normalization constant in the t -> 0 limit in the
                        K(t) definition for the connected part of the K(t)
                        normal_conn = <|\sum_n g(E_n)|**2>
            normal_uncon: float
                        Normalization constant in the t -> 0 limit in the
                        K(t) definition for the unconnected part of the K(t)
                        normal_unconn = |<\sum_n g(E_n)>|**2

        """

        filter_ = filter_func(spectra, *args, **kwargs)

        dims = spectra.shape[1]
        #  Effective dimensions of the Hilbert space
        #  after filtering
        dims_eff = np.mean(np.sum(filter_ ** 2, axis=1))

        #  Normalization constant at time = 0
        normal_con = np.mean(np.sum(filter_, axis=1) ** 2)

        #  Normalization constant at time = infty
        normal_uncon = np.mean(np.sum(filter_, axis=1))**2

        return filter_, dims, dims_eff, normal_con, normal_uncon

    def spectral_filtering(self,
                           filter_key='identity',
                           *args, **kwargs):
        """
        A function that returns a filter and appropriate
        normalization constrants used in filtering data
        before calculating the spectral form factor function.
        The class implementation of this function is a wrapper
        of the utils.misc module's function spectral_filtering()

        The spectral filter is a function of the form
            g(E_n), where E_n are the energies in the spectrum. An example
            of filter usage is the following calculation of the spectral form
            factor (SFF):

            K(t) = < |\sum_n g(E_n)\exp(-iE_n t)|**2>

            Parameters
            ----------
            unfolded: boolean
                        Whether filtering is performed on the unfolded
                        spectrum or not.
            filter_key: string
                        A key in the available_filters dict in the
                        utils.filter_functions module, whose
                        corresponding value is:
                        a function that accepts positional and keyword
                        arguments *args and **kwargs, respectively, and
                        returns a 2D ndarray of filtering values for
                        each single spectrum among spectra.


            Returns
            -------
            filter_: ndarray
                        A 2D ndarray, return of the filter_func function.
            dims: int
                        Hilbert space dimension of the input spectra before
                        filtering has been applied - spectra.shape[1]
            dims_eff: float
                        Effective dimensions of the Hilbert space after
                        filtering has been applied. Obtained by letting
                        t -> infty in the above K(t) definition and noting
                        that the square of the modulus can be written as a
                        double sum in which only the diagonal terms
                        contribute in the t -> infty limit. We then have:
                        dims_eff = <\sum_n (|g(E_n)|**2) >
            normal_con: float
                        Normalization constant in the t -> 0 limit in the
                        K(t) definition for the connected part of the K(t)
                        normal_conn = <|\sum_n g(E_n)|**2>
            normal_uncon: float
                        Normalization constant in the t -> 0 limit in the
                        K(t) definition for the unconnected part of the K(t)
                        normal_unconn = |<\sum_n g(E_n)>|**2

        """
        filt_dict = {}

        #  Issue a warning if the data are not for the unfolded/raw
        #  case

        try:
            filter_func = _fif.available_filters[filter_key]
        except KeyError:
            print(('Filtering option {} not yet implemented! '
                   'Calculations proceed with an identity filter.'
                   ).format(filter_key))
            filter_key = 'identity'
            filter_func = _fif.available_filters[filter_key]

        quantities = self._spectral_filtering(self.spectrum,
                                              filter_func, self._misc_dict,
                                              *args, **kwargs
                                              )

        dict_keys = ['filter', 'dims', 'dims_eff',
                     'normal_con', 'normal_uncon']

        for i, key in enumerate(dict_keys):
            filt_dict[key] = quantities[i]

        filt_dict['filter_type'] = filter_key

        # additional dict entries parsed from kwargs dict
        for (key, value) in iteritems(kwargs):
            filt_dict[key] = value

        self._filt_dict = filt_dict
        self._toggle_states(4)
