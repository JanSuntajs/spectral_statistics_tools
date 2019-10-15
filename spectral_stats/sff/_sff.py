import numpy as _np

from . import sff_functions as _sff


class Sff_mixin(object):

    def calc_sff(self, taulist, return_sfflist=False):
        """
        A function that calculates the sff on the
        postprocessed spectrum of eigenvalues.

        parameters:
        -----------

        taulist: ndarray

        a list of tau values at which sff is to be calculated

        return_sfflist: boolean, optional

        If True, the list of sff spectra for each individual
        energy spectrum is returned.
        """
        try:
            weights = self._filt_dict['filter']
        except KeyError:
            print(('Filtering has not yet been performed! '
                   'Calculating sff without a filter ...'
                   ))
            weights = _np.ones_like(self.spectrum, dtype=_np.float64)

        sfflist = _sff.sff_spectra(
            self.spectrum, taulist, weights)

        sff = _np.mean(_np.abs(sfflist)**2, axis=0)
        sff_err = _np.std(_np.abs(sfflist)**2, axis=0)

        mean = _np.abs(_np.mean(sfflist, axis=0))
        mean_err = _np.std(sfflist, axis=0) / mean

        sff_uncon = mean**2
        sff_uncon_err = 2 * mean_err * sff_uncon

        self.taulist = taulist
        self.sff = sff
        self.sff_err = sff_err
        self.sff_uncon = sff_uncon
        self.sff_uncon_err = sff_uncon_err
        self._toggle_states(5)

        if return_sfflist:

            return sfflist
