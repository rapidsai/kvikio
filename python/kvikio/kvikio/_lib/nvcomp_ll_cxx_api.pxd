# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

cdef extern from "cuda_runtime.h":
    ctypedef void* cudaStream_t

    ctypedef enum cudaMemcpyKind:
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4

cdef extern from "nvcomp.h":
    ctypedef enum nvcompType_t:
        NVCOMP_TYPE_CHAR = 0,       # 1B
        NVCOMP_TYPE_UCHAR = 1,      # 1B
        NVCOMP_TYPE_SHORT = 2,      # 2B
        NVCOMP_TYPE_USHORT = 3,     # 2B
        NVCOMP_TYPE_INT = 4,        # 4B
        NVCOMP_TYPE_UINT = 5,       # 4B
        NVCOMP_TYPE_LONGLONG = 6,   # 8B
        NVCOMP_TYPE_ULONGLONG = 7,  # 8B
        NVCOMP_TYPE_BITS = 0xff     # 1b

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

# nvCOMP Low-Level Interface.
# https://github.com/NVIDIA/nvcomp/blob/main/doc/lowlevel_c_quickstart.md

#
# LZ4 batch compression/decompression API.
#
cdef extern from "nvcomp/lz4.h" nogil:
    ctypedef struct nvcompBatchedLZ4Opts_t:
        nvcompType_t data_type

    cdef nvcompBatchedLZ4Opts_t nvcompBatchedLZ4DefaultOpts

    # Compression API.
    cdef nvcompStatus_t nvcompBatchedLZ4CompressGetTempSize(
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedLZ4Opts_t format_opts,
        size_t* temp_bytes
    )

    cdef nvcompStatus_t nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedLZ4Opts_t format_opts,
        size_t* max_compressed_bytes
    )

    cdef nvcompStatus_t nvcompBatchedLZ4CompressAsync(
        const void* const* device_uncompressed_ptrs,
        const size_t* device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_compressed_ptrs,
        size_t* device_compressed_bytes,
        nvcompBatchedLZ4Opts_t format_opts,
        cudaStream_t stream
    )

    # Decompression API.
    cdef nvcompStatus_t nvcompBatchedLZ4DecompressGetTempSize(
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes
    )

    cdef nvcompStatus_t nvcompBatchedLZ4GetDecompressSizeAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        size_t* device_uncompressed_bytes,
        size_t batch_size,
        cudaStream_t stream
    )

    nvcompStatus_t nvcompBatchedLZ4DecompressAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        const size_t* device_uncompressed_bytes,
        size_t* device_actual_uncompressed_bytes,
        size_t batch_size,
        void* const device_temp_ptr,
        size_t temp_bytes,
        void* const* device_uncompressed_ptrs,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream
    )

#
# Gdeflate batch compression/decompression API.
#
cdef extern from "nvcomp/gdeflate.h" nogil:
    ctypedef struct nvcompBatchedGdeflateOpts_t:
        int algo

    cdef nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateDefaultOpts

    # Compression API.
    cdef nvcompStatus_t nvcompBatchedGdeflateCompressGetTempSize(
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedGdeflateOpts_t format_opts,
        size_t* temp_bytes
    )

    cdef nvcompStatus_t nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedGdeflateOpts_t format_opts,
        size_t* max_compressed_bytes
    )

    cdef nvcompStatus_t nvcompBatchedGdeflateCompressAsync(
        const void* const* device_uncompressed_ptrs,
        const size_t* device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_compressed_ptrs,
        size_t* device_compressed_bytes,
        nvcompBatchedGdeflateOpts_t format_opts,
        cudaStream_t stream
    )

    # Decompression API.
    cdef nvcompStatus_t nvcompBatchedGdeflateDecompressGetTempSize(
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes
    )

    nvcompStatus_t nvcompBatchedGdeflateGetDecompressSizeAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        size_t* device_uncompressed_bytes,
        size_t batch_size,
        cudaStream_t stream
    )

    nvcompStatus_t nvcompBatchedGdeflateDecompressAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        const size_t* device_uncompressed_bytes,
        size_t* device_actual_uncompressed_bytes,
        size_t batch_size,
        void* const device_temp_ptr,
        size_t temp_bytes,
        void* const* device_uncompressed_ptrs,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream
    )

#
# zstd batch compression/decompression API.
#
cdef extern from "nvcomp/zstd.h" nogil:
    ctypedef struct nvcompBatchedZstdOpts_t:
        int reserved

    cdef nvcompBatchedZstdOpts_t nvcompBatchedZstdDefaultOpts

    # Compression API.
    cdef nvcompStatus_t nvcompBatchedZstdCompressGetTempSize(
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedZstdOpts_t format_opts,
        size_t* temp_bytes
    )

    cdef nvcompStatus_t nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedZstdOpts_t format_opts,
        size_t* max_compressed_bytes
    )

    cdef nvcompStatus_t nvcompBatchedZstdCompressAsync(
        const void* const* device_uncompressed_ptrs,
        const size_t* device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_compressed_ptrs,
        size_t* device_compressed_bytes,
        nvcompBatchedZstdOpts_t format_opts,
        cudaStream_t stream
    )

    # Decompression API.
    cdef nvcompStatus_t nvcompBatchedZstdDecompressGetTempSize(
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes
    )

    nvcompStatus_t nvcompBatchedZstdGetDecompressSizeAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        size_t* device_uncompressed_bytes,
        size_t batch_size,
        cudaStream_t stream
    )

    nvcompStatus_t nvcompBatchedZstdDecompressAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        const size_t* device_uncompressed_bytes,
        size_t* device_actual_uncompressed_bytes,
        size_t batch_size,
        void* const device_temp_ptr,
        size_t temp_bytes,
        void* const* device_uncompressed_ptrs,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream
    )

#
# Snappy batch compression/decompression API.
#
cdef extern from "nvcomp/snappy.h" nogil:
    ctypedef struct nvcompBatchedSnappyOpts_t:
        int reserved

    cdef nvcompBatchedSnappyOpts_t nvcompBatchedSnappyDefaultOpts

    # Compression API.
    cdef nvcompStatus_t nvcompBatchedSnappyCompressGetTempSize(
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedSnappyOpts_t format_opts,
        size_t* temp_bytes
    )

    cdef nvcompStatus_t nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedSnappyOpts_t format_opts,
        size_t* max_compressed_bytes
    )

    cdef nvcompStatus_t nvcompBatchedSnappyCompressAsync(
        const void* const* device_uncompressed_ptrs,
        const size_t* device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_compressed_ptrs,
        size_t* device_compressed_bytes,
        nvcompBatchedSnappyOpts_t format_opts,
        cudaStream_t stream
    )

    # Decompression API.
    cdef nvcompStatus_t nvcompBatchedSnappyDecompressGetTempSize(
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes
    )

    nvcompStatus_t nvcompBatchedSnappyGetDecompressSizeAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        size_t* device_uncompressed_bytes,
        size_t batch_size,
        cudaStream_t stream
    )

    nvcompStatus_t nvcompBatchedSnappyDecompressAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        const size_t* device_uncompressed_bytes,
        size_t* device_actual_uncompressed_bytes,
        size_t batch_size,
        void* const device_temp_ptr,
        size_t temp_bytes,
        void* const* device_uncompressed_ptrs,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream
    )


#
# Deflate batch compression/decompression API.
#
cdef extern from "nvcomp/deflate.h" nogil:
    ctypedef struct nvcompBatchedDeflateOpts_t:
        int algo

    cdef nvcompBatchedDeflateOpts_t nvcompBatchedDeflateDefaultOpts

    # Compression API.
    cdef nvcompStatus_t nvcompBatchedDeflateCompressGetTempSize(
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedDeflateOpts_t format_opts,
        size_t* temp_bytes
    )

    cdef nvcompStatus_t nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedDeflateOpts_t format_opts,
        size_t* max_compressed_bytes
    )

    cdef nvcompStatus_t nvcompBatchedDeflateCompressAsync(
        const void* const* device_uncompressed_ptrs,
        const size_t* device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes,
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_compressed_ptrs,
        size_t* device_compressed_bytes,
        nvcompBatchedDeflateOpts_t format_opts,
        cudaStream_t stream
    )

    # Decompression API.
    cdef nvcompStatus_t nvcompBatchedDeflateDecompressGetTempSize(
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes
    )

    nvcompStatus_t nvcompBatchedDeflateGetDecompressSizeAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        size_t* device_uncompressed_bytes,
        size_t batch_size,
        cudaStream_t stream
    )

    nvcompStatus_t nvcompBatchedDeflateDecompressAsync(
        const void* const* device_compressed_ptrs,
        const size_t* device_compressed_bytes,
        const size_t* device_uncompressed_bytes,
        size_t* device_actual_uncompressed_bytes,
        size_t batch_size,
        void* const device_temp_ptr,
        size_t temp_bytes,
        void* const* device_uncompressed_ptrs,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream
    )
