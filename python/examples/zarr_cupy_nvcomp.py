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

    # Normally, we cannot assume that GPU and CPU compressors are compatible.
    # E.g., `open_cupy_array()` uses nvCOMP's Snappy GPU compression by default,
    # which, as far as we know, isn’t compatible with any CPU compressor. Thus,
    # let's re-write our Zarr array using a CPU and GPU compatible compressor.
    z = kvikio.zarr.open_cupy_array(
        store=path,
        mode="w",
        shape=(20,),
        chunks=(5,),
        compressor=kvikio.zarr.CompatCompressor.lz4(),
    )
    z[:] = a

    # Because we are using a CompatCompressor, it is now possible to open the file
    # using Zarr's built-in LZ4 decompressor that uses the CPU.
    z = zarr.open_array(path)
    # `z` is now read as a regular NumPy array
    assert isinstance(z[:], numpy.ndarray)
    assert (a.get() == z[:]).all()
    # and we can write to is as usual
    z[:] = numpy.arange(20, 40)

    # And we can read the Zarr file back into a CuPy array.
    z = kvikio.zarr.open_cupy_array(store=path, mode="r")
    assert isinstance(z[:], cupy.ndarray)
    assert (cupy.arange(20, 40) == z[:]).all()


if __name__ == "__main__":
    main("/tmp/zarr-cupy-nvcomp")
