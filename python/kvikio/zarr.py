# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
from abc import abstractmethod
from typing import Any, Mapping, Sequence

import cupy
import numpy
import numpy as np
import zarr
import zarr.creation
import zarr.storage
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray_like
from numcodecs.registry import register_codec
from packaging.version import parse

import kvikio
import kvikio.nvcomp

MINIMUM_ZARR_VERSION = "2.15"

# Is this version of zarr supported? We depend on the `Context`
# argument introduced in https://github.com/zarr-developers/zarr-python/pull/1131
# in zarr 2.15.
supported = parse(zarr.__version) >= parse(MINIMUM_ZARR_VERSION)


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

    def __init__(self, *args, **kwargs) -> None:
        if not kvikio.zarr.supported:
            raise RuntimeError(
                f"GDSStore requires Zarr >={kvikio.zarr.MINIMUM_ZARR_VERSION}"
            )
        super().__init__(*args, **kwargs)

    def __eq__(self, other):
        return isinstance(other, GDSStore) and self.path == other.path

    def _tofile(self, a, fn):
        with kvikio.CuFile(fn, "w") as f:
            written = f.write(a)
            assert written == a.nbytes

    def getitems(
        self,
        keys: Sequence[str],
        *,
        contexts: Mapping[str, Mapping] = {},
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


class NVCompCompressor(Codec):
    """Abstract base class for nvCOMP compressors

    The derived classes must set `codec_id` and implement
    `get_nvcomp_manager`

    Parameters
    ----------
    device_ordinal
        The device that should do the compression/decompression
    """

    def __init__(self, device_ordinal: int = 0):
        self.device_ordinal = device_ordinal

    @abstractmethod
    def get_nvcomp_manager(self) -> kvikio.nvcomp.nvCompManager:
        """Abstract method that should return the nvCOMP compressor manager

        Returns
        -------
        nvCompManager
            The nvCOMP compressor manager to use
        """
        pass  # TODO: cache Manager

    def encode(self, buf) -> cupy.ndarray:
        """Compress using `get_nvcomp_manager()`

        Parameters
        ----------
        buf : buffer-like
            The buffer to compress. Accepts both host and device memory.

        Returns
        -------
        cupy.ndarray
            The compressed buffer wrapped in a CuPy array
        """
        buf = cupy.asarray(ensure_contiguous_ndarray_like(buf))
        return self.get_nvcomp_manager().compress(buf)

    def decode(self, buf, out=None):
        """Decompress using `get_nvcomp_manager()`

        Parameters
        ----------
        buf : buffer-like
            The buffer to decompress. Accepts both host and device memory.
        out : buffer-like, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data. Accepts both host and
            device memory.

        Returns
        -------
        buffer-like
            Decompress data, which is either host or device memory based on the type
            of `out`. If `out` is None, the type of `buf` determines the return buffer
            type.
        """
        buf = ensure_contiguous_ndarray_like(buf)
        is_host_buffer = not hasattr(buf, "__cuda_array_interface__")
        if is_host_buffer:
            buf = cupy.asarray(buf)

        ret = self.get_nvcomp_manager().decompress(buf)

        if is_host_buffer:
            ret = cupy.asnumpy(ret)

        if out is not None:
            out = ensure_contiguous_ndarray_like(out)
            if hasattr(out, "__cuda_array_interface__"):
                cupy.copyto(out, ret.view(dtype=out.dtype), casting="no")
            else:
                np.copyto(out, cupy.asnumpy(ret.view(dtype=out.dtype)), casting="no")
        return ret


class ANS(NVCompCompressor):
    codec_id = "nvcomp_ANS"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.ANSManager(device_id=self.device_ordinal)


class Bitcomp(NVCompCompressor):
    codec_id = "nvcomp_Bitcomp"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.BitcompManager(device_id=self.device_ordinal)


class Cascaded(NVCompCompressor):
    codec_id = "nvcomp_Cascaded"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.CascadedManager(device_id=self.device_ordinal)


class Gdeflate(NVCompCompressor):
    codec_id = "nvcomp_Gdeflate"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.GdeflateManager(device_id=self.device_ordinal)


class LZ4(NVCompCompressor):
    codec_id = "nvcomp_LZ4"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.LZ4Manager(device_id=self.device_ordinal)


class Snappy(NVCompCompressor):
    codec_id = "nvcomp_Snappy"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.SnappyManager(device_id=self.device_ordinal)


# Expose a list of available nvCOMP compressors and register them as Zarr condecs
nvcomp_compressors = [ANS, Bitcomp, Cascaded, Gdeflate, LZ4, Snappy]
for c in nvcomp_compressors:
    register_codec(c)
