import numpy as _np

from . import _sff_functions as _sff


class Sff_mixin(object):

    def sff(self, taulist):

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
        sff_uncon = _np.abs(_np.mean(sfflist, axis=0))**2

        return taulist, sff, sff_uncon
