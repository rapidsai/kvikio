# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
from typing import Any, Mapping, Sequence

import numpy
import zarr.storage

import kvikio


class GDSStore(zarr.storage.DirectoryStore):
    """GPUDirect Storage (GDS) class using directories and files.

    This class works like `zarr.storage.DirectoryStore` but implements
    getitems() in order to support direct reading into device memory.
    It uses KvikIO for reads and writes, which in turn will use GDS
    when applicable.

    Notes
    -----
    GDSStore doesn't implement `_fromfile()` thus non-array data such as
    meta data is always read into host memory.
    This is because only zarr.Array use getitems() to retrieve data.
    """

    def __eq__(self, other):
        return isinstance(other, GDSStore) and self.path == other.path

    def _tofile(self, a, fn):
        with kvikio.CuFile(fn, "w") as f:
            written = f.write(a)
            assert written == a.nbytes

    def getitems(
        self, keys: Sequence[str], contexts: Mapping[str, Mapping] = {}
    ) -> Mapping[str, Any]:

        default_meta_array = numpy.empty(())
        files = []
        ret = {}
        io_results = []
        try:
            for key in keys:
                filepath = os.path.join(self.path, key)
                if not os.path.isfile(filepath):
                    continue
                try:
                    meta_array = contexts[key]["meta_array"]
                except KeyError:
                    meta_array = default_meta_array

                nbytes = os.path.getsize(filepath)
                f = kvikio.CuFile(filepath, "r")
                files.append(f)
                ret[key] = numpy.empty_like(meta_array, shape=(nbytes,), dtype="u1")
                io_results.append((f.pread(ret[key]), nbytes))

            for future, nbytes in io_results:
                nbytes_read = future.get()
                if nbytes_read != nbytes:
                    raise RuntimeError(
                        f"Incomplete read ({nbytes_read}) expected {nbytes}"
                    )
        finally:
            for f in files:
                f.close()
        return ret
