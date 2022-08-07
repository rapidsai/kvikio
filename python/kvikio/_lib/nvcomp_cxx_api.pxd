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

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from libc.stdint cimport uint8_t, uint32_t

cdef extern from "cuda_runtime.h":
    ctypedef void* cudaStream_t

cdef extern from "nvcomp.h":
    ctypedef enum nvcompType_t:
        NVCOMP_TYPE_CHAR = 0,      # 1B
        NVCOMP_TYPE_UCHAR = 1,     # 1B
        NVCOMP_TYPE_SHORT = 2,     # 2B
        NVCOMP_TYPE_USHORT = 3,    # 2B
        NVCOMP_TYPE_INT = 4,       # 4B
        NVCOMP_TYPE_UINT = 5,      # 4B
        NVCOMP_TYPE_LONGLONG = 6,  # 8B
        NVCOMP_TYPE_ULONGLONG = 7, # 8B
        NVCOMP_TYPE_BITS = 0xff    # 1b


cdef extern from "nvcomp/shared_types.h":
    ctypedef enum nvcompStatus_t:
        nvcompSuccess = 0,
        nvcompErrorInvalidValue = 10,
        nvcompErrorNotSupported = 11,
        nvcompErrorCannotDecompress = 12,
        nvcompErrorBadChecksum = 13,
        nvcompErrorCannotVerifyChecksums = 14,
        nvcompErrorCudaError = 1000,
        nvcompErrorInternal = 10000,


# Manager Factory
cdef extern from "nvcomp/nvcompManagerFactory.hpp" namespace 'nvcomp':
    cdef shared_ptr[nvcompManagerBase] create_lz4_manager "nvcomp::create_manager"(
        const uint8_t* comp_buffer,
        cudaStream_t stream,
        const int device_id) except +


# Compresion Manager
cdef extern from "nvcomp/nvcompManager.hpp" namespace 'nvcomp':
    cdef cppclass PinnedPtrPool[T]:
        pass

    cdef cppclass CompressionConfig "nvcomp::CompressionConfig":
        const size_t uncompressed_buffer_size 
        const size_t max_uncompressed_buffer_size 
        const size_t num_chunks 
        CompressionConfig(
            PinnedPtrPool[nvcompStatus_t]* pool,
            size_t uncompressed_buffer_size) except +
        nvcompStatus_t* get_status() const
        CompressionConfig (CompressionConfig&& other) except +
        CompressionConfig (const CompressionConfig& other) except +
        CompressionConfig& operator= (CompressionConfig&& other) except +
        CompressionConfig& operator= (const CompressionConfig& other) except +

    cdef cppclass DecompressionConfig "nvcomp::DecompressionConfig":
        size_t decomp_data_size
        uint32_t num_chunks
        DecompressionConfig(PinnedPtrPool[nvcompStatus_t]& pool)
        nvcompStatus_t* get_status() const
        DecompressionConfig(DecompressionConfig&& other)
        DecompressionConfig(const DecompressionConfig& other)
        DecompressionConfig& operator=(DecompressionConfig&& other)
        DecompressionConfig& operator=(const DecompressionConfig& other)

    cdef cppclass nvcompManagerBase "nvcomp::nvcompManagerBase":
        CompressionConfig configure_compression (
            const size_t decomp_buffer_size)
        void compress(
            const uint8_t* decomp_buffer, 
            uint8_t* comp_buffer,
            const CompressionConfig& comp_config)
        DecompressionConfig configure_decompression (
            const uint8_t* comp_buffer)
        DecompressionConfig configure_decompression (
            const CompressionConfig& comp_config)
        void decompress(
            uint8_t* decomp_buffer, 
            const uint8_t* comp_buffer,
            const DecompressionConfig& decomp_config)
        void set_scratch_buffer(uint8_t* new_scratch_buffer)
        size_t get_required_scratch_buffer_size() except +
        size_t get_compressed_output_size(uint8_t* comp_buffer)

    cdef cppclass PimplManager "nvcomp::PimplManager":
        CompressionConfig configure_compression (
            const size_t decomp_buffer_size) except +
        void compress(
            const uint8_t* decomp_buffer, 
            uint8_t* comp_buffer,
            const CompressionConfig& comp_config) except +
        DecompressionConfig configure_decompression (
            const uint8_t* comp_buffer) except +
        DecompressionConfig configure_decompression (
            const CompressionConfig& comp_config) except +
        void decompress(
            uint8_t* decomp_buffer, 
            const uint8_t* comp_buffer,
            const DecompressionConfig& decomp_config) except +
        void set_scratch_buffer(uint8_t* new_scratch_buffer) except +
        size_t get_required_scratch_buffer_size() except +
        size_t get_compressed_output_size(uint8_t* comp_buffer) except +

# C++ Abstract LZ4 Manager
cdef extern from "nvcomp/lz4.hpp":
    cdef cppclass LZ4Manager "nvcomp::LZ4Manager":
        LZ4Manager (
            size_t uncomp_chunk_size,
            nvcompType_t data_type,
            cudaStream_t user_stream,
            const int device_id
        ) except +
        CompressionConfig configure_compression (
            const size_t decomp_buffer_size
        ) except +
        void compress(
            const uint8_t* decomp_buffer, 
            uint8_t* comp_buffer,
            const CompressionConfig& comp_config
        ) except +
        DecompressionConfig configure_decompression (
            const uint8_t* comp_buffer
        ) except +
        DecompressionConfig configure_decompression (
            const CompressionConfig& comp_config
        ) except +
        void decompress(
            uint8_t* decomp_buffer, 
            const uint8_t* comp_buffer,
            const DecompressionConfig& decomp_config
        ) except +
        void set_scratch_buffer(uint8_t* new_scratch_buffer) except +
        size_t get_required_scratch_buffer_size() except +
        size_t get_compressed_output_size(uint8_t* comp_buffer) except +

# Low-level LZ4 API
cdef extern from "nvcomp/lz4.h":

    cdef nvcompStatus_t _nvcompBatchedLZ4CompressGetTempSize \
        "nvcompBatchedLZ4CompressGetTempSize" (
            size_t batch_size,
            size_t max_uncompressed_chunk_bytes,
            size_t* temp_bytes)except+

    cdef nvcompStatus_t _nvcompBatchedLZ4CompressGetMaxOutputChunkSize \
        "nvcompBatchedLZ4CompressGetMaxOutputChunkSize" (
            size_t max_uncompressed_chunk_bytes,
            size_t* max_compressed_bytes)except+

    cdef nvcompStatus_t _nvcompBatchedLZ4CompressAsync "nvcompBatchedLZ4CompressAsync" (
        const void* const* device_in_ptr,
        const size_t* device_in_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_out_ptr,
        size_t* device_out_bytes,
        cudaStream_t stream)except+

    cdef nvcompStatus_t _nvcompBatchedLZ4DecompressAsync \
        "nvcompBatchedLZ4DecompressAsync" (
            const void* const* device_in_ptrs,
            const size_t* device_in_bytes,
            const size_t* device_out_bytes,  # unused
            size_t max_uncompressed_chunk_bytes,  # unused
            size_t batch_size,
            void* const device_temp_ptr,  # unused
            const size_t temp_bytes,  # unused
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
