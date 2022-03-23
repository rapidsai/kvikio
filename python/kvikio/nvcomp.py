# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import collections
from enum import Enum

import cupy as cp
import numpy as np

import kvikio._lib.pynvcomp as _lib


def make_ptr_collection(data: collections.abc.Collection) -> list:
    """ Returns a list of ndarray pointers.
    """
    return [_lib.__get_ptr(x) for x in data]


def get_ptr(x: np.ndarray) -> int:
    """ Returns the ndarray pointer for an NDArrayLike.
    """
    return _lib.__get_ptr(x)


def cp_to_nvcomp_dtype(in_type: cp.dtype) -> Enum:
    """ Returns `int` value of the NVCOMP_TYPE for supported dtypes.
    """
    cp_type = cp.dtype(in_type)
    return {
        cp.dtype("int8"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_CHAR,
        cp.dtype("uint8"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_UCHAR,
        cp.dtype("int16"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_SHORT,
        cp.dtype("uint16"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_USHORT,
        cp.dtype("int32"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_INT,
        cp.dtype("uint32"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_UINT,
        cp.dtype("int64"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_LONGLONG,
        cp.dtype("uint64"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_ULONGLONG,
    }[cp_type]


class CascadedOptions:
    """ Options to pass to the Cascaded Compressor.
    """

    def __init__(self, num_RLEs: int = 1, num_deltas: int = 1, use_bp: bool = True):
        self.num_RLEs = num_RLEs
        self.num_deltas = num_deltas
        self.use_bp = use_bp


class CascadedCompressor:
    def __init__(self, dtype: cp.dtype, config: CascadedOptions = CascadedOptions()):
        """ Initialize a CascadedCompressor and Decompressor for a
        specific dtype.
        """
        self.dtype = dtype
        self.config = config
        self.compressor = _lib._CascadedCompressor(
            cp_to_nvcomp_dtype(self.dtype).value,
            config.num_RLEs,
            config.num_deltas,
            config.use_bp,
        )
        self.decompressor = _lib._CascadedDecompressor()
        self.s = cp.cuda.Stream()

    def compress(self, data: cp.ndarray) -> cp.ndarray:
        # TODO: An option: check if incoming data size matches the size of the
        # last incoming data, and reuse temp and out buffer if so.
        data_size = data.size * data.itemsize
        self.compress_temp_size = np.zeros((1,), dtype=np.int64)
        self.compress_out_size = np.zeros((1,), dtype=np.int64)
        self.compressor.configure(
            data_size, self.compress_temp_size, self.compress_out_size
        )
        self.compress_temp_buffer = cp.zeros(self.compress_temp_size, dtype=np.uint8)
        self.compress_out_buffer = cp.zeros(self.compress_out_size, dtype=np.uint8)
        self.compressor.compress_async(
            data,
            data_size,
            self.compress_temp_buffer,
            self.compress_temp_size,
            self.compress_out_buffer,
            self.compress_out_size,
            self.s.ptr,
        )
        return self.compress_out_buffer[: self.compress_out_size[0]]

    def decompress(self, data: cp.ndarray) -> cp.ndarray:
        # TODO: logic to reuse temp buffer if it is large enough
        data_size = data.size * data.itemsize
        self.decompress_temp_size = np.zeros((1,), dtype=np.int64)
        self.decompress_out_size = np.zeros((1,), dtype=np.int64)

        self.decompressor.configure(
            data,
            data_size,
            self.decompress_temp_size,
            self.decompress_out_size,
            self.s.ptr,
        )

        self.decompress_temp_buffer = cp.zeros(
            self.decompress_temp_size, dtype=np.uint8
        )
        self.decompress_out_buffer = cp.zeros(self.decompress_out_size, dtype=np.uint8)
        self.decompressor.decompress_async(
            data,
            data_size,
            self.decompress_temp_buffer,
            self.decompress_temp_size,
            self.decompress_out_buffer,
            self.decompress_out_size,
            self.s.ptr,
        )
        return self.decompress_out_buffer.view(self.dtype)


class LZ4Compressor:
    def __init__(self, dtype: cp.dtype):
        self.dtype = dtype
        self.compressor = _lib._LZ4Compressor()
        self.decompressor = _lib._LZ4Decompressor()
        self.s = cp.cuda.Stream()

    def compress(self, data: cp.ndarray) -> cp.ndarray:
        # TODO: An option: check if incoming data size matches the size of the
        # last incoming data, and reuse temp and out buffer if so.
        data_size = data.size * data.itemsize
        self.compress_temp_size = np.zeros((1,), dtype=np.int64)
        self.compress_out_size = np.zeros((1,), dtype=np.int64)
        self.compressor.configure(
            data_size, self.compress_temp_size, self.compress_out_size
        )

        self.compress_temp_buffer = cp.zeros(
            (self.compress_temp_size[0],), dtype=cp.uint8
        )
        self.compress_out_buffer = cp.zeros(
            (self.compress_out_size[0],), dtype=cp.uint8
        )
        # Weird issue with LZ4 Compressor - if you pass it a gpu-side out_size
        # pointer it will error. If you pass it a host-side out_size pointer it will
        # segfault.
        self.gpu_out_size = cp.array(self.compress_out_size, dtype=np.int64)
        self.compressor.compress_async(
            data,
            data_size,
            self.compress_temp_buffer,
            self.compress_temp_size,
            self.compress_out_buffer,
            self.gpu_out_size,
            self.s.ptr,
        )
        return self.compress_out_buffer[: self.compress_out_size[0]]

    def decompress(self, data: cp.ndarray) -> cp.ndarray:
        # TODO: logic to reuse temp buffer if it is large enough
        data_size = data.size * data.itemsize
        self.decompress_temp_size = np.zeros((1,), dtype=np.int64)
        self.decompress_out_size = np.zeros((1,), dtype=np.int64)

        self.decompressor.configure(
            data,
            data_size,
            self.decompress_temp_size,
            self.decompress_out_size,
            self.s.ptr,
        )

        self.decompress_temp_buffer = cp.zeros(
            self.decompress_temp_size, dtype=np.uint8
        )
        self.decompress_out_buffer = cp.zeros(self.decompress_out_size, dtype=np.uint8)
        self.decompressor.decompress_async(
            data,
            data_size,
            self.decompress_temp_buffer,
            self.decompress_temp_size,
            self.decompress_out_buffer,
            self.decompress_out_size,
            self.s.ptr,
        )
        return self.decompress_out_buffer.view(self.dtype)


class SnappyCompressor:
    def __init__(self):
        self.compressor = _lib.LibSnappyCompressor()

    def compress(self, data: cp.ndarray) -> cp.ndarray:
        # convert data to pointers
        data_sizes = cp.array([256], dtype=cp.uint64)

        num_chunks = 1
        max_chunk_size = 256
        format_opts = 0
        # get output sizke
        max_compressed_chunk_size = np.zeros(1, dtype=np.uint64)
        self.compressor._get_compress_max_output_chunk_size(
            max_chunk_size, max_compressed_chunk_size, format_opts
        )
        temp_size = np.zeros(1, dtype=np.uint64)
        # get temp size
        result = self.compressor._get_compress_temp_size(
            num_chunks, max_compressed_chunk_size, temp_size, format_opts
        )

        # allocate output buffers
        output_buffers = cp.zeros(512, dtype=cp.int8)
        output_buffers_ptr = get_ptr(output_buffers)
        # call compress with sizes and buffers

        s = cp.cuda.Stream()

        data = cp.array(list(range(0, 256)), dtype=np.uint8)
        data_ptr = cp.array([get_ptr(data)], dtype=np.uint64)
        data_sizes = cp.array([256], dtype=np.uint64)
        num_chunks = cp.array([1], dtype=np.uint64)
        output_buffers = cp.zeros(1024, dtype=np.uint8)
        output_buffers_ptr = cp.array([get_ptr(output_buffers)], dtype=np.uint64)
        output_sizes = cp.zeros(1, dtype=np.uint64)
        format_opts = 0

        result = self.compressor._compress(
            data_ptr,
            data_sizes,
            np.array([0]),
            num_chunks,
            0,
            0,
            output_buffers_ptr,
            output_sizes,
            format_opts,
            s.ptr,
        )

        return result
