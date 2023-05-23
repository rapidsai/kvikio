# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import h5py
import numpy as np

import legate_kvikio.kerchunk


def hdf5_io(filename):
    a = np.arange(10000).reshape((100, 100))

    # Write array using h5py
    with h5py.File(filename, "w") as f:
        f.create_dataset("mydataset", chunks=(10, 10), data=a)

    # Read hdf5 file using legate+kerchunk
    b = legate_kvikio.kerchunk.hdf5_read(filename, dataset_name="mydataset")

    # They should be equal
    assert (a == b).all()


if __name__ == "__main__":
    hdf5_io("/tmp/legate-kvikio-io.hdf5")
