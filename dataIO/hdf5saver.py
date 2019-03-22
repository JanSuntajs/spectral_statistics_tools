"""
This module implements functionality for storing
the calculated data in a HDF5 format which allows
for structured filesaving and addition of metadata.

"""

from future.utils import iteritems
import h5py


def hdf5save(filename, datasets,
             attrs, *args, **kwargs):
    """

    Parameters
    ----------
    filename: string
                Filename string without
                the .hdf5 suffix specifying
                the name under which the
                hdf5 file will be saved.
    datasets: dict
                A dict of key and
                value pairs where keys
                are data descriptor strings
                and values are the actual
                numerical values, prefferably
                in the numpy array format.
    attrs: dict
                A (nested) dist of key and
                value pairs; keys of the
                attrs dict should be a subset
                of the datasets' keys. The values
                are dictionaries of values which
                should be appended as datasets'
                attributes. Large arrays should
                preferably not be appended as
                attributes but rather as
                datasets.



    Returns
    -------


    """

    filename = filename.strip() + '.hdf5'
    with h5py.File(filename, 'w') as f:

        for (key, value) in iteritems(datasets):
            f.create_dataset(key, data=value)
            try:
                f[key].attrs.update(attrs[key])
            except KeyError:
                print(("Key {} in attrs "
                       "dict missing!").format(key))
