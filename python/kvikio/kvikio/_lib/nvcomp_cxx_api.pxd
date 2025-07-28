# Copyright (c) 2022 Carson Swope
# Use, modification, and distribution is subject to the MIT License
# https://github.com/carsonswope/py-nvcomp/blob/main/LICENSE)
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
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

from libc.stdint cimport uint8_t, uint32_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector


cdef extern from "cuda_runtime.h":
    ctypedef void* cudaStream_t

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

# Manager Factory
cdef extern from "nvcomp/nvcompManagerFactory.hpp" namespace 'nvcomp':
    cdef shared_ptr[nvcompManagerBase] create_manager "nvcomp::create_manager"(
        const uint8_t* comp_buffer
    ) except +


# Compression Manager
cdef extern from "nvcomp/nvcompManager.hpp" namespace 'nvcomp':
    cdef cppclass PinnedPtrPool[T]:
        pass

    cdef cppclass CompressionConfig "nvcomp::CompressionConfig":
        const size_t uncompressed_buffer_size
        const size_t max_compressed_buffer_size
        const size_t num_chunks
        CompressionConfig(
            PinnedPtrPool[nvcompStatus_t]* pool,
            size_t uncompressed_buffer_size) except +
        nvcompStatus_t* get_status() const
        CompressionConfig(CompressionConfig& other)
        CompressionConfig& operator=(const CompressionConfig& other) except +
        # Commented as Cython doesn't support rvalues, but a user can call
        # `move` with the existing operator and generate correct C++ code
        # xref: https://github.com/cython/cython/issues/1445
        # CompressionConfig& operator=(CompressionConfig&& other) except +

    cdef cppclass DecompressionConfig "nvcomp::DecompressionConfig":
        size_t decomp_data_size
        uint32_t num_chunks
        DecompressionConfig(PinnedPtrPool[nvcompStatus_t]& pool) except +
        nvcompStatus_t* get_status() const
        DecompressionConfig(DecompressionConfig& other)
        DecompressionConfig& operator=(const DecompressionConfig& other) except +
        # Commented as Cython doesn't support rvalues, but a user can call
        # `move` with the existing operator and generate correct C++ code
        # xref: https://github.com/cython/cython/issues/1445
        # DecompressionConfig& operator=(DecompressionConfig&& other) except +

    cdef cppclass nvcompManagerBase "nvcomp::nvcompManagerBase":
        CompressionConfig configure_compression(
            const size_t decomp_buffer_size)
        void compress(
            const uint8_t* decomp_buffer,
            uint8_t* comp_buffer,
            const CompressionConfig& comp_config) except +
        DecompressionConfig configure_decompression(
            const uint8_t* comp_buffer)
        DecompressionConfig configure_decompression(
            const CompressionConfig& comp_config)
        void decompress(
            uint8_t* decomp_buffer,
            const uint8_t* comp_buffer,
            const DecompressionConfig& decomp_config)
        size_t get_compressed_output_size(uint8_t* comp_buffer) except +

    cdef cppclass PimplManager "nvcomp::PimplManager":
        CompressionConfig configure_compression(
            const size_t decomp_buffer_size) except +
        void compress(
            const uint8_t* decomp_buffer,
            uint8_t* comp_buffer,
            const CompressionConfig& comp_config) except +
        DecompressionConfig configure_decompression(
            const uint8_t* comp_buffer)
        DecompressionConfig configure_decompression(
            const CompressionConfig& comp_config)
        void decompress(
            uint8_t* decomp_buffer,
            const uint8_t* comp_buffer,
            const DecompressionConfig& decomp_config) except +
        size_t get_compressed_output_size(uint8_t* comp_buffer) except +

# C++ Concrete ANS Manager
cdef extern from "nvcomp/ans.h" nogil:
    ctypedef enum nvcompANSType_t:
        nvcomp_rANS = 0

    ctypedef struct nvcompBatchedANSOpts_t:
        nvcompANSType_t type
    cdef nvcompBatchedANSOpts_t nvcompBatchedANSDefaultOpts

cdef extern from "nvcomp/ans.hpp":
    cdef cppclass ANSManager "nvcomp::ANSManager":
        ANSManager(
            size_t uncomp_chunk_size,
            const nvcompBatchedANSOpts_t& format_opts,
        ) except +

# C++ Concrete Bitcomp Manager
cdef extern from "nvcomp/bitcomp.h" nogil:
    ctypedef struct nvcompBatchedBitcompFormatOpts:
        int algorithm_type
        nvcompType_t data_type
    cdef nvcompBatchedBitcompFormatOpts nvcompBatchedBitcompDefaultOpts

cdef extern from "nvcomp/bitcomp.hpp":
    cdef cppclass BitcompManager "nvcomp::BitcompManager":
        BitcompManager(
            size_t uncomp_chunk_size,
            const nvcompBatchedBitcompFormatOpts& format_opts,
        ) except +

# C++ Concrete Cascaded Manager
cdef extern from "nvcomp/cascaded.h" nogil:
    ctypedef struct nvcompBatchedCascadedOpts_t:
        size_t chunk_size
        nvcompType_t type
        int num_RLEs
        int num_deltas
        int use_bp
    cdef nvcompBatchedCascadedOpts_t nvcompBatchedCascadedDefaultOpts

cdef extern from "nvcomp/cascaded.hpp" nogil:
    cdef cppclass CascadedManager "nvcomp::CascadedManager":
        CascadedManager(
            size_t uncomp_chunk_size,
            const nvcompBatchedCascadedOpts_t& options,
        )

# C++ Concrete Gdeflate Manager
cdef extern from "nvcomp/gdeflate.h" nogil:
    ctypedef struct nvcompBatchedGdeflateOpts_t:
        int algo
    cdef nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateDefaultOpts

cdef extern from "nvcomp/gdeflate.hpp":
    cdef cppclass GdeflateManager "nvcomp::GdeflateManager":
        GdeflateManager(
            int uncomp_chunk_size,
            const nvcompBatchedGdeflateOpts_t& format_opts,
        ) except +

# C++ Concrete LZ4 Manager
cdef extern from "nvcomp/gdeflate.h" nogil:
    ctypedef struct nvcompBatchedLZ4Opts_t:
        nvcompType_t data_type
    cdef nvcompBatchedLZ4Opts_t nvcompBatchedLZ4DefaultOpts

cdef extern from "nvcomp/lz4.hpp":
    cdef cppclass LZ4Manager "nvcomp::LZ4Manager":
        LZ4Manager(
            size_t uncomp_chunk_size,
            const nvcompBatchedLZ4Opts_t& format_opts,
        ) except +

# C++ Concrete Snappy Manager
cdef extern from "nvcomp/snappy.h" nogil:
    ctypedef struct nvcompBatchedSnappyOpts_t:
        int reserved
    cdef nvcompBatchedSnappyOpts_t nvcompBatchedSnappyDefaultOpts

cdef extern from "nvcomp/snappy.hpp":
    cdef cppclass SnappyManager "nvcomp::SnappyManager":
        SnappyManager(
            size_t uncomp_chunk_size,
            const nvcompBatchedSnappyOpts_t& format_opts,
        ) except +
