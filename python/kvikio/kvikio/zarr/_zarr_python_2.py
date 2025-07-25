# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
from __future__ import annotations

import contextlib
import os
import os.path
from abc import abstractmethod
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import cupy
import cupy.typing
import numcodecs
import numpy
import numpy as np
import zarr
import zarr.creation
import zarr.errors
import zarr.storage
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray_like
from numcodecs.registry import register_codec
from packaging.version import parse

import kvikio
import kvikio._nvcomp
import kvikio._nvcomp_codec
import kvikio.zarr
from kvikio._nvcomp_codec import NvCompBatchCodec
from kvikio.numcodecs import BufferLike, CudaCodec

MINIMUM_ZARR_VERSION = "2.15"

# Is this version of zarr supported? We depend on the `Context`
# argument introduced in https://github.com/zarr-developers/zarr-python/pull/1131
# in zarr v2.15.
supported = parse(zarr.__version__) >= parse(MINIMUM_ZARR_VERSION)


class GDSStore(zarr.storage.DirectoryStore):  # type: ignore[name-defined]
    """GPUDirect Storage (GDS) class using directories and files.

    This class works like `zarr.storage.DirectoryStore` but implements
    getitems() in order to support direct reading into device memory.
    It uses KvikIO for reads and writes, which in turn will use GDS
    when applicable.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    compressor_config_overwrite
        If not None, use this `Mapping` to specify what is written to the Zarr metadata
        file on disk (`.zarray`). Normally, Zarr writes the configuration[1] given by
        the `compressor` argument to the `.zarray` file. Use this argument to overwrite
        the normal configuration and use the specified `Mapping` instead.
    decompressor_config_overwrite
        If not None, use this `Mapping` to specify what compressor configuration[1] is
        used for decompressing no matter the configuration found in the Zarr metadata
        on disk (the `.zarray` file).

    [1] https://github.com/zarr-developers/numcodecs/blob/cb155432/numcodecs/abc.py#L79

    Notes
    -----
    Atomic writes are used, which means that data are first written to a
    temporary file, then moved into place when the write is successfully
    completed. Files are only held open while they are being read or written and are
    closed immediately afterwards, so there is no need to manually close any files.

    Safe to write in multiple threads or processes.
    """

    # The default output array type used by getitems().
    default_meta_array = numpy.empty(())

    def __init__(
        self,
        path,
        normalize_keys=False,
        dimension_separator=None,
        *,
        compressor_config_overwrite: Optional[Mapping] = None,
        decompressor_config_overwrite: Optional[Mapping] = None,
    ) -> None:
        if not kvikio.zarr.supported:
            raise RuntimeError(
                f"GDSStore requires Zarr >={kvikio.zarr.MINIMUM_ZARR_VERSION}"
            )
        super().__init__(
            path, normalize_keys=normalize_keys, dimension_separator=dimension_separator
        )
        self.compressor_config_overwrite = compressor_config_overwrite
        self.decompressor_config_overwrite = decompressor_config_overwrite

    def __eq__(self, other):
        return isinstance(other, GDSStore) and self.path == other.path

    def _tofile(self, a, fn):
        with kvikio.CuFile(fn, "w") as f:
            written = f.write(a)
            assert written == a.nbytes

    def __getitem__(self, key):
        ret = super().__getitem__(key)
        if self.decompressor_config_overwrite and key == ".zarray":
            meta = self._metadata_class.decode_array_metadata(ret)
            if meta["compressor"]:
                meta["compressor"] = self.decompressor_config_overwrite
                ret = self._metadata_class.encode_array_metadata(meta)
        return ret

    def __setitem__(self, key, value):
        if self.compressor_config_overwrite and key == ".zarray":
            meta = self._metadata_class.decode_array_metadata(value)
            if meta["compressor"]:
                meta["compressor"] = self.compressor_config_overwrite
                value = self._metadata_class.encode_array_metadata(meta)
        super().__setitem__(key, value)

    def getitems(
        self,
        keys: Sequence[str],
        *,
        contexts: Mapping[str, Mapping] = {},
    ) -> Mapping[str, Any]:
        """Retrieve data from multiple keys.

        Parameters
        ----------
        keys : Iterable[str]
            The keys to retrieve
        contexts: Mapping[str, Context]
            A mapping of keys to their context. Each context is a mapping of store
            specific information. If the "meta_array" key exist, GDSStore use its
            values as the output array otherwise GDSStore.default_meta_array is used.

        Returns
        -------
        Mapping
            A collection mapping the input keys to their results.
        """
        ret = {}
        io_results = []

        with contextlib.ExitStack() as stack:
            for key in keys:
                filepath = os.path.join(self.path, key)
                if not os.path.isfile(filepath):
                    continue
                try:
                    meta_array = contexts[key]["meta_array"]
                except KeyError:
                    meta_array = self.default_meta_array

                nbytes = os.path.getsize(filepath)
                f = stack.enter_context(kvikio.CuFile(filepath, "r"))
                ret[key] = numpy.empty_like(meta_array, shape=(nbytes,), dtype="u1")
                io_results.append((f.pread(ret[key]), nbytes))

            for future, nbytes in io_results:
                nbytes_read = future.get()
                if nbytes_read != nbytes:
                    raise RuntimeError(
                        f"Incomplete read ({nbytes_read}) expected {nbytes}"
                    )
        return ret


class NVCompCompressor(CudaCodec):
    """Abstract base class for nvCOMP compressors

    The derived classes must set `codec_id` and implement
    `get_nvcomp_manager`
    """

    @abstractmethod
    def get_nvcomp_manager(self) -> kvikio.nvcomp.nvCompManager:
        """Abstract method that should return the nvCOMP compressor manager"""
        pass  # TODO: cache Manager

    def encode(self, buf: BufferLike) -> cupy.typing.NDArray:
        buf = cupy.asarray(ensure_contiguous_ndarray_like(buf))
        return self.get_nvcomp_manager().compress(buf)

    def decode(self, buf: BufferLike, out: Optional[BufferLike] = None) -> BufferLike:
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
        return kvikio.nvcomp.ANSManager()


class Bitcomp(NVCompCompressor):
    codec_id = "nvcomp_Bitcomp"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.BitcompManager()


class Cascaded(NVCompCompressor):
    codec_id = "nvcomp_Cascaded"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.CascadedManager()


class Gdeflate(NVCompCompressor):
    codec_id = "nvcomp_Gdeflate"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.GdeflateManager()


class LZ4(NVCompCompressor):
    codec_id = "nvcomp_LZ4"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.LZ4Manager()


class Snappy(NVCompCompressor):
    codec_id = "nvcomp_Snappy"

    def get_nvcomp_manager(self):
        return kvikio.nvcomp.SnappyManager()


# Expose a list of available nvCOMP compressors and register them as Zarr condecs
nvcomp_compressors = [ANS, Bitcomp, Cascaded, Gdeflate, LZ4, Snappy]
for c in nvcomp_compressors:
    register_codec(c)


class CompatCompressor:
    """A pair of compatible compressors one using the CPU and one using the GPU

    Warning
    -------
    `CompatCompressor` is only supported by KvikIO's `open_cupy_array()` and
    cannot be used as a compressor argument in Zarr functions like `open()`
    and `open_array()` directly. However, it is possible to use its `.cpu`
    like: `open(..., compressor=CompatCompressor.lz4().cpu)`.

    Parameters
    ----------
    cpu
        The CPU compressor.
    gpu
        The GPU compressor.
    """

    def __init__(self, cpu: Codec, gpu: CudaCodec) -> None:
        self.cpu = cpu
        self.gpu = gpu

    @classmethod
    def lz4(cls) -> CompatCompressor:
        """A compatible pair of LZ4 compressors"""
        return cls(cpu=numcodecs.LZ4(), gpu=NvCompBatchCodec("lz4"))


def open_cupy_array(
    store: Union[os.PathLike, str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    compressor: Codec | CompatCompressor = Snappy(),
    meta_array=cupy.empty(()),
    **kwargs,
) -> zarr.Array:
    """Open an Zarr array as a CuPy-like array using file-mode-like semantics.

    This function is a CUDA friendly version of `zarr.open_array` that reads
    and writes to CuPy arrays. Beside the arguments listed below, the arguments
    have the same semantics as in `zarr.open_array`.

    Parameters
    ----------
    store
        Path to directory in file system. As opposed to `zarr.open_array`,
        Store and path to zip files isn't supported.
    mode
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    compressor
        The compressor used when creating a Zarr file or None if no compressor
        is to be used. If a `CompatCompressor` is given, `CompatCompressor.gpu`
        is used for compression and decompression; and `CompatCompressor.cpu`
        is written as the compressor in the Zarr file metadata on disk.
        This argument is ignored in "r" and "r+" mode. By default the
        Snappy compressor by nvCOMP is used.
    meta_array : array-like, optional
        An CuPy-like array instance to use for determining arrays to create and
        return to users. It must implement `__cuda_array_interface__`.
    **kwargs
        The rest of the arguments are forwarded to `zarr.open_array` as-is.

    Returns
    -------
    Zarr array backed by a GDS file store, nvCOMP compression, and CuPy arrays.
    """

    if not isinstance(store, (str, os.PathLike)):
        raise ValueError("store must be a path")
    store = str(os.fspath(store))
    if not hasattr(meta_array, "__cuda_array_interface__"):
        raise ValueError("meta_array must implement __cuda_array_interface__")

    if mode in ("r", "r+", "a"):
        # In order to handle "a", we start by trying to open the file in read mode.
        try:
            ret = zarr.open_array(
                store=kvikio.zarr.GDSStore(path=store),  # type: ignore[call-arg]
                mode="r+",
                meta_array=meta_array,
                **kwargs,
            )
        except (
            zarr.errors.ContainsGroupError,
            zarr.errors.ArrayNotFoundError,  # type: ignore[attr-defined]
        ):
            # If we are reading, this is a genuine error.
            if mode in ("r", "r+"):
                raise
        else:
            if ret.compressor is None:
                return ret
            # If we are reading a LZ4-CPU compressed file, we overwrite the
            # metadata on-the-fly to make Zarr use LZ4-GPU for both compression
            # and decompression.
            compat_lz4 = CompatCompressor.lz4()
            if ret.compressor == compat_lz4.cpu:
                ret = zarr.open_array(
                    store=kvikio.zarr.GDSStore(  # type: ignore[call-arg]
                        path=store,
                        compressor_config_overwrite=compat_lz4.cpu.get_config(),
                        decompressor_config_overwrite=compat_lz4.gpu.get_config(),
                    ),
                    mode=mode,
                    meta_array=meta_array,
                    **kwargs,
                )
            elif not isinstance(ret.compressor, CudaCodec):
                raise ValueError(
                    "The Zarr file was written using a non-CUDA compatible "
                    f"compressor, {ret.compressor}, please use something "
                    "like kvikio.zarr.CompatCompressor"
                )
            return ret

    # At this point, we known that we are writing a new array
    if mode not in ("w", "w-", "a"):
        raise ValueError(f"Unknown mode: {mode}")

    if isinstance(compressor, CompatCompressor):
        compressor_config_overwrite = compressor.cpu.get_config()
        decompressor_config_overwrite = compressor.gpu.get_config()
        compressor = compressor.gpu
    else:
        compressor_config_overwrite = None
        decompressor_config_overwrite = None

    return zarr.open_array(
        store=kvikio.zarr.GDSStore(  # type: ignore[call-arg]
            path=store,
            compressor_config_overwrite=compressor_config_overwrite,
            decompressor_config_overwrite=decompressor_config_overwrite,
        ),
        mode=mode,
        meta_array=meta_array,
        compressor=compressor,
        **kwargs,
    )
