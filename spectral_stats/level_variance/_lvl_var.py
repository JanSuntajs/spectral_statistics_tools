

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

        lvl_var = _lvl.sigma_averaged(lvals, self.spectrum, deviation)

        self.lvals = lvals
        self.lvl_var = lvl_var
        self._toggle_states(6)
