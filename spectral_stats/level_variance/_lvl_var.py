import numpy as _np

from . import lvl_var_functions as _lvl


class Lvl_mixin(object):

    def calc_lvl_var(self, lvals, deviation=False):
        """
        A method for calculating the number level
        variance of the energy spectrum.spectrum

        Parameters
        ----------

        lvals: ndarray
                        1D ndarray of energy levels
                        for which the level variance
                        is to be computed.
        deviation: boolean
                        If True, also the standard
                        deviation of the averaged data
                        is returned.

        """

        #     = _lvl.sigma_averaged(
        #         self.spectrum, lvals, deviation)

        # sff = _np.mean(_np.abs(sfflist)**2, axis=0)
        # sff_uncon = _np.abs(_np.mean(sfflist, axis=0))**2

        # self.taulist = taulist
        # self.sff = sff
        # self.sff_uncon = sff_uncon
        # self._toggle_states(5)
        pass