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
from libc.stdint cimport int32_t, uint8_t, uintptr_t
from libcpp cimport bool, nullptr
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move

from kvikio._lib.arr cimport Array
from kvikio._lib.nvcomp_cxx_api cimport (
    ANSManager,
    CascadedManager,
    CompressionConfig,
    DecompressionConfig,
    LZ4Manager,
    SnappyManager,
    create_manager,
    cudaStream_t,
    nvcompBatchedCascadedDefaultOpts,
    nvcompBatchedCascadedOpts_t,
    nvcompManagerBase,
    nvcompStatus_t,
    nvcompType_t,
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
    # Temporary storage for factory allocated manager to prevent cleanup
    cdef shared_ptr[nvcompManagerBase] _mgr
    cdef nvcompManagerBase* _impl
    cdef shared_ptr[CompressionConfig] _compression_config
    cdef shared_ptr[DecompressionConfig] _decompression_config

    def configure_compression(self, decomp_buffer_size):
        cdef shared_ptr[CompressionConfig] partial = make_shared[CompressionConfig](
            self._impl.configure_compression(decomp_buffer_size)
        )
        self._compression_config = make_shared[CompressionConfig](
            (move(partial.get()[0]))
        )
        return {
            "uncompressed_buffer_size": self._compression_config.get()[
                0
            ].uncompressed_buffer_size,
            "max_compressed_buffer_size": self._compression_config.get()[
                0
            ].max_compressed_buffer_size,
            "num_chunks": self._compression_config.get()[0].num_chunks
        }

    def compress(self, decomp_buffer, comp_buffer):
        self._impl.compress(
            <const uint8_t*><uintptr_t>decomp_buffer.data.ptr,
            <uint8_t*><uintptr_t>comp_buffer.data.ptr,
            <CompressionConfig&>self._compression_config.get()[0]
        )
        size = self._impl.get_compressed_output_size(
            <uint8_t*><uintptr_t>comp_buffer.data.ptr
        )
        return size

    def configure_decompression_with_compressed_buffer(
        self,
        comp_buffer
    ) -> dict:
        cdef shared_ptr[DecompressionConfig] partial = make_shared[
            DecompressionConfig](self._impl.configure_decompression(
                <uint8_t*><uintptr_t>comp_buffer.data.ptr
            )
        )
        self._decompression_config = make_shared[DecompressionConfig](
            (move(partial.get()[0]))
        )
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
            <uint8_t*><uintptr_t>decomp_buffer.data.ptr,
            <const uint8_t*><uintptr_t>comp_buffer.data.ptr,
            <DecompressionConfig&>self._decompression_config.get()[0]
        )

    def set_scratch_buffer(self, new_scratch_buffer):
        return self._impl.set_scratch_buffer(
            <uint8_t*><uintptr_t>new_scratch_buffer.data.ptr
        )

    def get_required_scratch_buffer_size(self):
        return self._impl.get_required_scratch_buffer_size()

    def get_compressed_output_size(self, comp_buffer):
        return self._impl.get_compressed_output_size(
            <uint8_t*><uintptr_t>comp_buffer.data.ptr
        )


cdef class _LZ4Manager(_nvcompManager):
    def __cinit__(
        self,
        size_t uncomp_chunk_size,
        nvcompType_t data_type,
        user_stream,
        const int device_id,
    ):
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
    def __cinit__(
        self,
        size_t uncomp_chunk_size,
        user_stream,
        const int device_id,
    ):
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

cdef class _ANSManager(_nvcompManager):
    def __cinit__(
        self,
        size_t uncomp_chunk_size,
        user_stream,
        const int device_id,
    ):
        self._impl = <nvcompManagerBase*>new ANSManager(
            uncomp_chunk_size,
            <cudaStream_t><void*>0,  # TODO
            device_id
        )


cdef class _ManagedManager(_nvcompManager):
    def __init__(self, compressed_buffer):
        cdef shared_ptr[nvcompManagerBase] _mgr = create_manager(
            <uint8_t*><uintptr_t>compressed_buffer.data.ptr
        )
        self._mgr = _mgr
        self._impl = move(_mgr).get()
