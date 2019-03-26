"""
This module implements functionality for storing
the calculated data in a HDF5 format which allows
for structured filesaving and addition of metadata.

"""

from future.utils import iteritems
import h5py
import json

# special datasets are supposed to be stored as
# dicts using json.dumps
_special_sets = ['misc', 'metadata', 'system_info']


def hdf5save(filename, datasets,
             attrs, metadata={}, system_info={},
             *args, **kwargs):
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
                A dict of key and
                value pairs which provide
                additional information about
                the saved datasets. These are
                appended as the attributes of
                the whole group.

    Returns
    -------

    hdf5 file with:
        - datasets, which are accesed as
        f[key] where key is a key in the
        dataset dict
        Additional datasets are:
        - 'metadata', 'misc' and
        'system_info'
    """

    filename = filename.strip() + '.hdf5'
    with h5py.File(filename, 'w') as f:

        for (key, value) in iteritems(datasets):
            f.create_dataset(key, data=value)

        # add additional info about the
        # numerical procedure
        f.create_dataset('misc',
                         data=json.dumps(attrs))

        # add metadata
        f.create_dataset('metadata', data=json.dumps(metadata))

        # add info about the (physical) system
        f.create_dataset('system_info',
                         data=json.dumps(system_info))


def hdf5names(filename):
    """
    A routine that returns a dictionary
    of the hdf5 files' datasets.
    """
    try:
        with h5py.File(filename, 'w') as f:
            for key in f.keys():
                print(key)
    except OSError:
        print('{} not present!'.format(filename))


def hdf5load(filename, partial=False, keys={}, *args, **kwargs):
    """

    Parameters
    ----------

    filename: string
                Filename string without the
                .hdf5 suffix specifying the
                name under which the hdf5
                file will be saved.
    partial: boolean
                If True, only a part of the
                dataset is actually loaded
                into memory.
    keys: list
                If partial==True, this
                list specifies which
                datasets are to be loaded.

    """

    filename = filename.strip('.hdf5') + '.hdf5'

    return_dict = {}

    with h5py.File(filename, 'r') as f:
        if partial:
            keylist = keys
        else:
            keylist = f.keys()
        for key in keylist:
            if key not in _special_sets:
                return_dict[key] = f[key][:]
            else:
                return_dict[key] = json.loads(f[key][()])

    return return_dict
