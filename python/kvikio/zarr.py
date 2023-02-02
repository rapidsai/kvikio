# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path

import cupy
import zarr.storage

import kvikio
from kvikio._lib.arr import asarray


class GDSStore(zarr.storage.DirectoryStore):
    """GPUDirect Storage (GDS) class using directories and files.

    This class works like `zarr.storage.DirectoryStore` but use GPU
    buffers and will use GDS when applicable.
    The store supports both CPU and GPU buffers but when reading, GPU
    buffers are returned always.

    TODO: Write metadata to disk in order to preserve the item types such that
    GPU items are read as GPU device buffers and CPU items are read as bytes.
    """

    def __eq__(self, other):
        return isinstance(other, GDSStore) and self.path == other.path

    def _fromfile(self, fn):
        """Read `fn` into device memory _unless_ `fn` refers to Zarr metadata"""
        if os.path.basename(fn) in [
            zarr.storage.array_meta_key,
            zarr.storage.group_meta_key,
            zarr.storage.attrs_key,
        ]:
            return super()._fromfile(fn)
        else:
            nbytes = os.path.getsize(fn)
            with kvikio.CuFile(fn, "r") as f:
                ret = cupy.empty(nbytes, dtype="u1")
                read = f.read(ret)
                assert read == nbytes
                return ret

    def _tofile(self, a, fn):
        a = asarray(a)
        assert a.contiguous
        if a.cuda:
            with kvikio.CuFile(fn, "w") as f:
                written = f.write(a)
                assert written == a.nbytes
        else:
            super()._tofile(a.obj, fn)
