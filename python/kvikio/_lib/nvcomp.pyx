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

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from kvikio._lib.nvcomp cimport(
    __CascadedCompressor,
    __CascadedDecompressor,
    __LZ4Compressor,
    __LZ4Decompressor,
    cudaStream_t,
    nvcompBatchedSnappyCompressAsync,
    nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
    nvcompBatchedSnappyCompressGetTempSize,
    nvcompBatchedSnappyDecompressAsync,
    nvcompBatchedSnappyDecompressGetTempSize,
    nvcompBatchedSnappyOpts_t,
    nvcompStatus_t,
    nvcompType_t,
)


cpdef __get_array_interface_ptr(a):
    return a.__array_interface__['data'][0]

cpdef __get_cuda_array_interface_ptr(a):
    return a.__cuda_array_interface__['data'][0]

# could be either __array_interface__ or __cuda_array_interface__
cpdef __get_ptr(a):
    # this has to be slow though... is there a better way? try/catch?
    d = a.__dir__()
    if '__cuda_array_interface__' in d:
        return __get_cuda_array_interface_ptr(a)
    elif '__array_interface__' in d:
        return __get_array_interface_ptr(a)
    else:
        raise AttributeError('Argument does not implement __cuda_array_interface__ or __array_interface__')  # NOQA: E501


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


# _Cascaded Compressor / Decompressor
cdef class _CascadedCompressor:
    cdef __CascadedCompressor* c

    def __cinit__(self, nvcompType_t t, int num_RLEs, int num_deltas, bool use_bp):
        self.c = new __CascadedCompressor(
            t,
            num_RLEs,
            num_deltas,
            use_bp
        )

    def __dealloc__(self):
        del self.c

    def configure(self, in_bytes, temp_bytes, out_bytes):
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
        cdef size_t temp_bytes_val = (<size_t*>temp_bytes_ptr)[0]
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        cdef size_t out_bytes_val = (<size_t*>out_bytes_ptr)[0]
        self.c.configure(
            in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr)

    def compress_async(
        self,
        in_arr,
        in_bytes,
        temp_arr,
        temp_bytes,
        out_arr,
        out_bytes,
        uintptr_t stream=0
    ):
        cdef uintptr_t in_ptr=__get_ptr(in_arr)
        cdef uintptr_t temp_ptr=__get_ptr(temp_arr)
        cdef uintptr_t out_ptr=__get_ptr(out_arr)
        cdef uintptr_t out_bytes_ptr=__get_ptr(out_bytes)
        self.c.compress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

cdef class _CascadedDecompressor:
    cdef __CascadedDecompressor* d

    def __cinit__(self):
        self.d = new __CascadedDecompressor()

    def __dealloc__(self):
        del self.d

    cpdef configure(self, in_arr, in_bytes, temp_bytes, out_bytes, uintptr_t stream=0):
        cdef uintptr_t in_ptr = __get_ptr(in_arr)
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.d.configure(
            <void*>in_ptr,
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

    def decompress_async(
        self,
        in_arr,
        in_bytes,
        temp_arr,
        temp_bytes,
        out_arr,
        out_bytes,
        uintptr_t stream=0
    ):
        cdef uintptr_t in_ptr = __get_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_ptr(out_arr)
        self.d.decompress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t>out_bytes,
            <cudaStream_t>stream)


# LZ4 Compressor
cdef extern from "nvcomp/lz4.hpp" namespace 'nvcomp':
    cdef cppclass __LZ4Compressor "nvcomp::LZ4Compressor":
        __LZ4Compressor() except+

        void configure(
            const size_t in_bytes,
            size_t* temp_bytes,
            size_t* out_bytes) except+

        void compress_async(
            const void* in_ptr,
            const size_t in_bytes,
            void* temp_ptr,
            const size_t temp_bytes,
            void* out_ptr,
            size_t* out_bytes,
            cudaStream_t stream) except+

    cdef cppclass __LZ4Decompressor "nvcomp::LZ4Decompressor":
        __LZ4Decompressor() except+

        void configure(
            const void* in_ptr,
            const size_t in_bytes,
            size_t* temp_bytes,
            size_t* out_bytes,
            cudaStream_t stream) except+

        void decompress_async(
            const void* in_ptr,
            const size_t in_bytes,
            void* temp_ptr,
            const size_t temp_bytes,
            void* out_ptr,
            const size_t out_bytes,
            cudaStream_t stream) except+

# LZ4 Compressor / Decompressor
cdef class _LZ4Compressor:
    cdef __LZ4Compressor* c

    def __cinit__(self, size_t chunk_size=0):
        self.c = new __LZ4Compressor()

    def __dealloc__(self):
        del self.c

    def configure(self, in_bytes, temp_bytes, out_bytes):
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.c.configure(
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr)

    def compress_async(
        self,
        in_arr,
        in_bytes,
        temp_arr,
        temp_bytes,
        out_arr,
        out_bytes,
        uintptr_t stream=0
    ):
        cdef uintptr_t in_ptr = __get_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_ptr(out_arr)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.c.compress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

cdef class _LZ4Decompressor:
    cdef __LZ4Decompressor* d

    def __cinit__(self):
        self.d = new __LZ4Decompressor()

    def __dealloc__(self):
        del self.d

    cpdef configure(self, in_arr, in_bytes, temp_bytes, out_bytes, uintptr_t stream=0):
        cdef uintptr_t in_ptr = __get_ptr(in_arr)
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.d.configure(
            <void*>in_ptr,
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

    def decompress_async(
        self,
        in_arr,
        in_bytes,
        temp_arr,
        temp_bytes,
        out_arr,
        out_bytes,
        uintptr_t stream=0
    ):
        cdef uintptr_t in_ptr = __get_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_ptr(out_arr)
        self.d.decompress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t>out_bytes,
            <cudaStream_t>stream)


class _LibSnappyCompressor:
    def _get_decompress_temp_size(
        self,
        num_chunks,
        max_uncompressed_chunk_size,
        temp_bytes
    ):
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
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
        cdef uintptr_t device_compressed_bytes_ptr = __get_ptr(device_compressed_bytes)
        cdef uintptr_t device_uncompressed_bytes_ptr = __get_ptr(
            device_uncompressed_bytes
        )
        cdef uintptr_t device_actual_uncompressed_bytes_ptr = __get_ptr(
            device_actual_uncompressed_bytes
        )
        cdef uintptr_t device_statuses_ptr = __get_ptr(device_statuses)
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
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
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
        cdef uintptr_t max_compressed_size_ptr = __get_ptr(max_compressed_size)
        cdef nvcompBatchedSnappyOpts_t opts
        opts.reserved = format_opts
        print('ptr: ', max_compressed_size_ptr)
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
        cdef uintptr_t device_uncompressed_buffers_ptr = __get_ptr(
            device_uncompressed_buffers
        )
        cdef uintptr_t device_uncompressed_sizes_ptr = __get_ptr(
            device_uncompressed_sizes
        )
        cdef uintptr_t max_uncompressed_chunk_size_ptr = __get_ptr(
            max_uncompressed_chunk_size
        )
        cdef uintptr_t device_temp_buffer_ptr = 0
        cdef uintptr_t device_compressed_buffers_ptr = __get_ptr(
            device_compressed_buffers
        )
        cdef uintptr_t device_compressed_sizes_ptr = __get_ptr(
            device_compressed_sizes
        )
        cdef nvcompBatchedSnappyOpts_t opts
        opts.reserved = format_opts

        print('')
        print('device_uncompressed_ptr')
        print(hex(device_uncompressed_buffers_ptr))
        print('device_uncompressed_bytes')
        print(hex(device_uncompressed_sizes_ptr))
        print('batch size')
        cdef uintptr_t batch_size_ptr = __get_ptr(batch_size)
        print(hex(batch_size_ptr))
        print('device_compressed_ptr')
        print(hex(device_compressed_buffers_ptr))
        print('device_compressed_bytes')
        print(hex(device_compressed_sizes_ptr))
        print(hex(stream))

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
