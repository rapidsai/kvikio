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

cdef extern from "cuda_runtime.h":
    ctypedef void* cudaStream_t

cdef extern from "nvcomp.h":
    ctypedef enum nvcompStatus_t:
        nvcompSuccess = 0,
        nvcompErrorInvalidValue = 10
        nvcompErrorNotSupported = 11,
        nvcompErrorCannotDecompress = 12,
        nvcompErrorCudaError = 1000,
        nvcompErrorInternal = 10000

    ctypedef enum nvcompType_t 'nvcompType_t':
        NVCOMP_TYPE_CHAR,
        NVCOMP_TYPE_UCHAR,
        NVCOMP_TYPE_SHORT,
        NVCOMP_TYPE_USHORT,
        NVCOMP_TYPE_INT,
        NVCOMP_TYPE_UINT,
        NVCOMP_TYPE_LONGLONG,
        NVCOMP_TYPE_ULONGLONG,
        NVCOMP_TYPE_BITS
    
    ctypedef enum nvcompError_t:
        nvcompSuccess_ = 0,
        nvcompErrorInvalidValue_ = 10,
        nvcompErrorNotSupported_ = 11,
        nvcompErrorCudaError_ = 1000,
        nvcompErrorInternal_ = 10000

cdef enum pyNvcompType_t:
    pyNVCOMP_TYPE_CHAR = NVCOMP_TYPE_CHAR
    pyNVCOMP_TYPE_UCHAR = NVCOMP_TYPE_UCHAR
    pyNVCOMP_TYPE_SHORT = NVCOMP_TYPE_SHORT
    pyNVCOMP_TYPE_USHORT = NVCOMP_TYPE_USHORT
    pyNVCOMP_TYPE_INT = NVCOMP_TYPE_INT
    pyNVCOMP_TYPE_UINT = NVCOMP_TYPE_UINT
    pyNVCOMP_TYPE_LONGLONG = NVCOMP_TYPE_LONGLONG
    pyNVCOMP_TYPE_ULONGLONG = NVCOMP_TYPE_ULONGLONG
    pyNVCOMP_TYPE_BITS = NVCOMP_TYPE_BITS

# Cascaded Compressor
cdef extern from "nvcomp/cascaded.hpp" namespace 'nvcomp':
    cdef cppclass _CascadedCompressor "nvcomp::CascadedCompressor":
        _CascadedCompressor(nvcompType_t, int, int, bool) except+

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

    cdef cppclass _CascadedDecompressor "nvcomp::CascadedDecompressor":
        _CascadedDecompressor() except+

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

# LZ4 Compressor
cdef extern from "nvcomp/lz4.hpp" namespace 'nvcomp':
    cdef cppclass _LZ4Compressor "nvcomp::LZ4Compressor":
        _LZ4Compressor() except+

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

    cdef cppclass _LZ4Decompressor "nvcomp::LZ4Decompressor":
        _LZ4Decompressor() except+

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

# Low-level LZ4 API
cdef extern from "nvcomp/lz4.h":

    # _nvcompError_t
    cdef nvcompError_t _nvcompBatchedLZ4CompressGetTempSize "nvcompBatchedLZ4CompressGetTempSize" (
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes)except+

    cdef nvcompError_t _nvcompBatchedLZ4CompressGetMaxOutputChunkSize "nvcompBatchedLZ4CompressGetMaxOutputChunkSize" (
        size_t max_uncompressed_chunk_bytes,
        size_t* max_compressed_bytes)except+

    cdef nvcompError_t _nvcompBatchedLZ4CompressAsync "nvcompBatchedLZ4CompressAsync" (
        const void* const* device_in_ptr,
        const size_t* device_in_bytes,
        size_t max_uncompressed_chunk_bytes, # unused
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_out_ptr,
        size_t* device_out_bytes,
        cudaStream_t stream)except+

    cdef nvcompError_t _nvcompBatchedLZ4DecompressAsync "nvcompBatchedLZ4DecompressAsync" (
        const void* const* device_in_ptrs,
        const size_t* device_in_bytes,
        const size_t*  device_out_bytes, # unused
        size_t max_uncompressed_chunk_bytes, # unused
        size_t batch_size,
        void* const device_temp_ptr, # unused
        const size_t temp_bytes, # unused
        void* const* device_out_ptr,
        cudaStream_t stream) except+

# Snappy Compressor
cdef extern from "nvcomp/snappy.h" nogil:
    ctypedef struct nvcompBatchedSnappyOpts_t:
        int reserved

    cdef nvcompStatus_t nvcompBatchedSnappyDecompressGetTempSize(
        size_t num_chunks,
        size_t max_uncompressed_chunk_size,
        size_t* temp_bytes) except+

    cdef nvcompStatus_t nvcompBatchedSnappyDecompressAsync(
        void* device_compressed_ptrs,
        size_t* device_compressed_bytes,
        size_t* device_uncompressed_bytes,
        size_t* device_actual_uncompressed_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* device_uncompressed_ptr,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream) except+
    
    cdef nvcompStatus_t nvcompBatchedSnappyCompressGetTempSize(
        size_t batch_size,
        size_t max_chunk_size,
        nvcompBatchedSnappyOpts_t format_opts,
        size_t* temp_bytes_ptr) except+

    cdef nvcompStatus_t nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
        size_t max_chunk_size,
        nvcompBatchedSnappyOpts_t format_opts,
       size_t* max_compressed_size) except+

    cdef nvcompStatus_t nvcompBatchedSnappyCompressAsync(
        const void* const* device_uncompressed_ptr,
        const size_t* device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_compressed_ptr,
        size_t* device_compressed_bytes,
        nvcompBatchedSnappyOpts_t format_opts,
        cudaStream_t stream) except+
