# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cupy
import numpy
import zarr

import kvikio
import kvikio.zarr


def main(path):
    a = cupy.arange(20)

    # Let's use KvikIO's convenience function `open_cupy_array()` to create
    # a new Zarr file on disk. Its semantic is the same as `zarr.open_array()`
    # but uses a GDS file store, nvCOMP compression, and CuPy arrays.
    z = kvikio.zarr.open_cupy_array(store=path, mode="w", shape=(20,), chunks=(5,))

    # `z` is a regular Zarr Array that we can write to as usual
    z[0:10] = numpy.arange(0, 10)
    # but it also support direct reads and writes of CuPy arrays
    z[10:20] = cupy.arange(10, 20)

    # Reading `z` returns a CuPy array
    assert isinstance(z[:], cupy.ndarray)
    assert (a == z[:]).all()

    # By default, `open_cupy_array()` uses nvCOMP's `lz4` GPU compression, which is
    # compatible with NumCodecs's `lz4` CPU compression (CPU). Normally, it is not
    # possible to change which decompressor to use when reading a Zarr file. The
    # decompressor specified in the Zarr file's metadata is always used. However,
    # `open_cupy_array()` makes it possible to overwrite the metadata on-the-fly
    # without having to modify the Zarr file on disk. In fact, the Zarr file written
    # above appears, in the metadata, as if it was written by NumCodecs's `lz4` CPU
    # compression. Thus, we can open the file using Zarr's regular API and the CPU.
    z = zarr.open_array(path)
    # `z` is now read as a regular NumPy array
    assert isinstance(z[:], numpy.ndarray)
    assert (a.get() == z[:]).all()
    # and we can write to is as usual
    z[:] = numpy.arange(20, 40)

    # Let's read the Zarr file back into a CuPy array. Notice, even though the metadata
    # on disk is specifying NumCodecs's `lz4` CPU decompressor, `open_cupy_array` will
    # use nvCOMP to decompress the files.
    z = kvikio.zarr.open_cupy_array(store=path, mode="r")
    assert isinstance(z[:], cupy.ndarray)
    assert (cupy.arange(20, 40) == z[:]).all()


if __name__ == "__main__":
    main("/tmp/zarr-cupy-nvcomp")
