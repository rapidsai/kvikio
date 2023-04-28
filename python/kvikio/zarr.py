# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
from abc import abstractmethod

import cupy
import numpy as np
import zarr.creation
import zarr.storage
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray_like
from numcodecs.registry import register_codec

import kvikio
import kvikio.nvcomp
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
