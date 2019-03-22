"""
This module implements functionality for storing
the calculated data in a HDF5 format which allows
for structured filesaving and addition of metadata.

"""

from future.utils import iteritems
import h5py
import json


def hdf5save(filename, datasets,
             attrs, metadata={}, *args, **kwargs):
    """

    Parameters
    ----------
    filename: string
                Filename string without
                the .hdf5 suffix specifying
                the name under which the
                hdf5 file will be saved.
    groupname: string
                Name string specifying
                the name of the basegroup
                for which the datasets are
                created.
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


    """

    filename = filename.strip() + '.hdf5'
    with h5py.File(filename, 'w') as f:

        for (key, value) in iteritems(datasets):
            f.create_dataset(key, data=value)

        # add metadata
        f.create_dataset('misc',
                         data=json.dumps(attrs))

        f.create_dataset('metadata', data=json.dumps(metadata))


def hdf5load(filename):

    filename = filename.strip() + '.hdf5'

    return_dict = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            if key not in ['misc', 'metadata']:
                return_dict[key] = f[key][:]
            else:
                return_dict[key] = json.loads(f[key][()])

    return return_dict
