from utils._unfolding import unfold_mixin


class Checker_mixin(unfold_mixin):

    def _unfold_checker(self, unfolded):
        """
        A routine for making sure that the quantities
        calculated for the unfolded spectra are indeed
        calculated for proper spectral widthts etc.
        """

        if unfolded:
            #  If current instance's spectral width is not equal to the
            #  spectral width of the existing unfolded spectrum for
            #  which the quantities are to be calculated, print the
            #  warning message and call the spectral unfolding_function
            #  to perform unfolding again
            try:
                if self.spectral_width != self.unfolded['spectral_width']:
                    print(('Spectral width of the unfolded spectrum {} '
                           'does not match the instance\'s current '
                           'spectral width {}. '
                           'Calling spectral_unfolding() with the '
                           'current spectral width ...'
                           ).format(self.unfolded['spectral_width'],
                                    self.spectral_width))

                    n = self.unfolded['n']
                    merge = self.unfolded['merged']
                    correct = self.unfolded['correct_slope']

                    self.spectral_unfolding(n, merge, correct)
            #  If unfolding has not yet been performed, the above call
            #  would result in a TypeError as None object has no items
            #  to be accessed. In this case, print the warning message
            #  and call the spectral_unfolding procedure with default
            #  parameters and slope correction.
            except TypeError:
                # self.unfolded
                print(('Calling spectral_unfolding() with default parameters '
                       'and slope correction.'))

                self.spectral_unfolding(correct_slope=True)
        else:
            pass
