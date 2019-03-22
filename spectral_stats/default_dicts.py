"""
A module that stores default dictionary
structures

TO DO:

CONSIDER IF IT IS OK TO IMPLEMENT THIS
OR DOES IT BREAK SOME FUNCTIONALITY
CURRENTLY, THIS IS NOT YET IMPLEMENTED
"""
from future.utils import iteritems

import numpy as _np

#  unfold_dict default values
default_unfold_dict = {

    'n': 0,
    'merged': False,
    'correct_slope': False,
    'spectral_width': (0., 1.),
    'discarded': 0,

}

#  misc_dict default values
default_misc_dict = {
    'unfolded': False,
    'individual': False,
    'spectral_width': (0., 1.),
    'n': 0,
    'merged': False,
    'correct_slope': False,
    'mean_ener': 0.,
    'sq_ham_tr_sq': 0.,
    'sq_ham_tr_sq': 0.,
    'gamma': 0.,

}

#  filt dict
default_filt_dict = {

    'filter': _np.ndarray([]),
    'dims': 0,
    'dims_eff': 0,
    'normal_con': 1.,
    'normal_uncon': 1.,

}



