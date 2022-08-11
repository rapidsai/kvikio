# Copyright (c) 2022 Carson Swope
# Use, modification, and distribution is subject to the MIT License
# https://github.com/carsonswope/py-nvcomp/blob/main/LICENSE)
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from enum import Enum

import cupy as cp

import cython
from cython.operator cimport dereference
from libc.stdint cimport uintptr_t, uint8_t, int32_t
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.utility cimport move
from libcpp cimport bool, nullptr

from kvikio._lib.arr cimport Array
from kvikio._lib.nvcomp_cxx_api cimport (
    cudaStream_t,
    nvcompType_t,
    nvcompStatus_t,
    nvcompBatchedSnappyCompressAsync,
    nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
    nvcompBatchedSnappyCompressGetTempSize,
    nvcompBatchedSnappyDecompressAsync,
    nvcompBatchedSnappyDecompressGetTempSize,
    nvcompBatchedSnappyOpts_t,
    nvcompManagerBase,
    LZ4Manager,
    SnappyManager,
    CascadedManager,
    CompressionConfig,
    DecompressionConfig,
    nvcompLZ4FormatOpts,
    nvcompBatchedCascadedOpts_t,
    nvcompBatchedCascadedDefaultOpts,
    nvcompBatchedLZ4Opts_t,
    nvcompBatchedLZ4DefaultOpts,
    nvcompBatchedLZ4CompressGetTempSize,
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
    nvcompBatchedLZ4CompressAsync,
    nvcompBatchedLZ4DecompressGetTempSize,
    nvcompBatchedLZ4DecompressGetTempSizeEx,
    nvcompBatchedLZ4DecompressAsync,
    nvcompBatchedLZ4GetDecompressSizeAsync
)


class pyNvcompType_t(Enum):
    pyNVCOMP_TYPE_CHAR = nvcompType_t.NVCOMP_TYPE_CHAR
    pyNVCOMP_TYPE_UCHAR = nvcompType_t.NVCOMP_TYPE_UCHAR
    pyNVCOMP_TYPE_SHORT = nvcompType_t.NVCOMP_TYPE_SHORT
    pyNVCOMP_TYPE_USHORT = nvcompType_t.NVCOMP_TYPE_USHORT
    pyNVCOMP_TYPE_INT = nvcompType_t.NVCOMP_TYPE_INT
    pyNVCOMP_TYPE_UINT = nvcompType_t.NVCOMP_TYPE_UINT
    pyNVCOMP_TYPE_LONGLONG = nvcompType_t.NVCOMP_TYPE_LONGLONG
    pyNVCOMP_TYPE_ULONGLONG = nvcompType_t.NVCOMP_TYPE_ULONGLONG
    pyNVCOMP_TYPE_BITS = nvcompType_t.NVCOMP_TYPE_BITS

cdef class _nvcompManager:
    cdef nvcompManagerBase* _impl
    cdef shared_ptr[CompressionConfig] _compression_config
    cdef shared_ptr[DecompressionConfig] _decompression_config

    def configure_compression(self, decomp_buffer_size):
        cdef shared_ptr[CompressionConfig] partial = make_shared[CompressionConfig](
            self._impl.configure_compression(decomp_buffer_size)
        )
        self._compression_config = make_shared[CompressionConfig]((move(partial.get()[0])))
        return {
            "uncompressed_buffer_size": self._compression_config.get()[0].uncompressed_buffer_size,
            "max_compressed_buffer_size": self._compression_config.get()[0].max_compressed_buffer_size,
            "num_chunks": self._compression_config.get()[0].num_chunks
        }

    def compress(self, decomp_buffer, comp_buffer):
        self._impl.compress(
            <const uint8_t*><size_t>decomp_buffer.data.ptr,
            <uint8_t*><size_t>comp_buffer.data.ptr,
            <CompressionConfig&>self._compression_config.get()[0]
        )
        size = self._impl.get_compressed_output_size(<uint8_t*><size_t>comp_buffer.data.ptr)
        return size

    def configure_decompression_with_compressed_buffer(
        self,
        comp_buffer
    ) :
        cdef shared_ptr[DecompressionConfig] partial = make_shared[DecompressionConfig](
            self._impl.configure_decompression(<uint8_t*><size_t>comp_buffer.data.ptr)
        )
        self._decompression_config = make_shared[DecompressionConfig]((move(partial.get()[0])))
        return {
            "decomp_data_size": self._decompression_config.get()[0].decomp_data_size,
            "num_chunks": self._decompression_config.get()[0].num_chunks
        }
    
    def decompress(
        self,
        decomp_buffer, 
        comp_buffer,
    ):
        self._impl.decompress(
            <uint8_t*><size_t>decomp_buffer.data.ptr,
            <const uint8_t*><size_t>comp_buffer.data.ptr,
            <DecompressionConfig&>self._decompression_config.get()[0]
        )

    def set_scratch_buffer(self, new_scratch_buffer):
        return self._impl.set_scratch_buffer(<uint8_t*><size_t>new_scratch_buffer.data.ptr)

    def get_required_scratch_buffer_size(self):
        return self._impl.get_required_scratch_buffer_size()

    def get_compressed_output_size(self, comp_buffer):
        return self._impl.get_compressed_output_size(<uint8_t*><size_t>comp_buffer.data.ptr)


cdef class _LZ4Manager(_nvcompManager):
    def __cinit__(self,
        size_t uncomp_chunk_size,
        nvcompType_t data_type,
        user_stream,
        const int device_id,
    ):
        # print a pointer
        # print("{0:x}".format(<unsigned long>), var)
        # TODO: Doesn't work with user specified streams passed down
        # from anywhere up. I'm not going to rabbit hole on it until
        # everything else works.
        cdef cudaStream_t stream = <cudaStream_t><void*>user_stream
        self._impl = <nvcompManagerBase*>new LZ4Manager(
            uncomp_chunk_size,
            <nvcompType_t>data_type,
            <cudaStream_t><void*>0,  # TODO
            device_id
        )


cdef class _SnappyManager(_nvcompManager):
    def __cinit__(self,
        size_t uncomp_chunk_size,
        user_stream,
        const int device_id,
    ):
        # print a pointer
        # print("{0:x}".format(<unsigned long>), var)
        # TODO: Doesn't work with user specified streams passed down
        # from anywhere up. I'm not going to rabbit hole on it until
        # everything else works.
        self._impl = <nvcompManagerBase*>new SnappyManager(
            uncomp_chunk_size,
            <cudaStream_t><void*>0,  # TODO
            device_id
        )

cdef class _CascadedManager(_nvcompManager):
    def __cinit__(
        self,
        _options,
        user_stream,
        const int device_id,
    ):
        self._impl = <nvcompManagerBase*>new CascadedManager(
            <nvcompBatchedCascadedOpts_t>nvcompBatchedCascadedDefaultOpts,  # TODO
            <cudaStream_t><void*>0,  # TODO
            device_id,
        )

class _LibSnappyCompressor:
    def _get_decompress_temp_size(
        self,
        num_chunks,
        max_uncompressed_chunk_size,
        temp_bytes
    ):
        cdef uintptr_t temp_bytes_ptr = Array(temp_bytes).ptr
        return nvcompBatchedSnappyDecompressGetTempSize(
            <size_t>num_chunks,
            <size_t>max_uncompressed_chunk_size,
            <size_t*>temp_bytes_ptr
        )

    def _decompress(
        self,
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        device_actual_uncompressed_bytes,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        device_uncompressed_ptr,
        device_statuses,
        stream
    ):
        cdef uintptr_t device_compressed_bytes_ptr = Array(device_compressed_bytes).ptr
        cdef uintptr_t device_uncompressed_bytes_ptr = Array(
            device_uncompressed_bytes
        ).ptr
        cdef uintptr_t device_actual_uncompressed_bytes_ptr = Array(
            device_actual_uncompressed_bytes
        ).ptr
        cdef uintptr_t device_statuses_ptr = Array(device_statuses).ptr
        return nvcompBatchedSnappyDecompressAsync(
            <const void* const*><void*>device_compressed_ptrs,
            <size_t*>device_compressed_bytes_ptr,
            <size_t*>device_uncompressed_bytes_ptr,
            <size_t*>device_actual_uncompressed_bytes_ptr,
            <size_t>batch_size,
            <void*>device_temp_ptr,
            <size_t>temp_bytes,
            <void* const*><void*>device_uncompressed_ptr,
            <nvcompStatus_t*>device_statuses_ptr,
            <cudaStream_t>stream
        )

    def _get_compress_temp_size(
        self,
        batch_size,
        max_chunk_size,
        temp_bytes,
        format_opts
    ):
        cdef uintptr_t temp_bytes_ptr = Array(temp_bytes).ptr
        cdef nvcompBatchedSnappyOpts_t opts
        opts.reserved = format_opts
        return nvcompBatchedSnappyCompressGetTempSize(
            <size_t>batch_size,
            <size_t>max_chunk_size,
            <nvcompBatchedSnappyOpts_t>opts,
            <size_t*>temp_bytes_ptr
        )

    def _get_compress_max_output_chunk_size(
        self,
        max_chunk_size,
        max_compressed_size,
        format_opts
    ):
        cdef uintptr_t max_compressed_size_ptr = Array(max_compressed_size).ptr
        cdef nvcompBatchedSnappyOpts_t opts
        opts.reserved = format_opts
        return nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
            <size_t>max_chunk_size,
            <nvcompBatchedSnappyOpts_t>opts,
            <size_t*>max_compressed_size_ptr
        )

    def _compress(
        self,
        device_uncompressed_buffers,
        device_uncompressed_sizes,
        max_uncompressed_chunk_size,
        batch_size,
        device_temp_buffer,
        temp_size,
        device_compressed_buffers,
        device_compressed_sizes,
        format_opts,
        stream
    ):
        cdef uintptr_t device_uncompressed_buffers_ptr = Array(
            device_uncompressed_buffers
        ).ptr
        cdef uintptr_t device_uncompressed_sizes_ptr = Array(
            device_uncompressed_sizes
        ).ptr
        cdef uintptr_t max_uncompressed_chunk_size_ptr = Array(
            max_uncompressed_chunk_size
        ).ptr
        cdef uintptr_t device_temp_buffer_ptr = 0
        cdef uintptr_t device_compressed_buffers_ptr = Array(
            device_compressed_buffers
        ).ptr
        cdef uintptr_t device_compressed_sizes_ptr = Array(
            device_compressed_sizes
        ).ptr
        cdef nvcompBatchedSnappyOpts_t opts
        opts.reserved = format_opts

        cdef uintptr_t batch_size_ptr = Array(batch_size).ptr

        with nogil:
            result = nvcompBatchedSnappyCompressAsync(
                <const void* const*><void*>device_uncompressed_buffers_ptr,
                <const size_t*>device_uncompressed_sizes_ptr,
                <size_t>max_uncompressed_chunk_size_ptr,
                <size_t>batch_size_ptr,
                <void*>device_temp_buffer_ptr,
                <size_t>0,
                <void* const*><void*>device_compressed_buffers_ptr,
                <size_t*>device_compressed_sizes_ptr,
                <nvcompBatchedSnappyOpts_t>opts,
                <cudaStream_t>stream
            )

        return result