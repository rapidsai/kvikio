# Copyright (c) 2022 Carson Swope
# Use, modification, and distribution is subject to the MIT License
# https://github.com/carsonswope/py-nvcomp/blob/main/LICENSE)
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
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

from libc.stdint cimport uint8_t, uintptr_t
from libcpp cimport nullptr
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move

from kvikio._lib.arr cimport Array
from kvikio._lib.nvcomp_cxx_api cimport (
    ANSManager,
    BitcompManager,
    CascadedManager,
    CompressionConfig,
    DecompressionConfig,
    GdeflateManager,
    LZ4Manager,
    SnappyManager,
    create_manager,
    cudaStream_t,
    nvcompBatchedANSDefaultOpts,
    nvcompBatchedANSOpts_t,
    nvcompBatchedBitcompDefaultOpts,
    nvcompBatchedBitcompFormatOpts,
    nvcompBatchedCascadedDefaultOpts,
    nvcompBatchedCascadedOpts_t,
    nvcompBatchedGdeflateDefaultOpts,
    nvcompBatchedGdeflateOpts_t,
    nvcompBatchedLZ4DefaultOpts,
    nvcompBatchedLZ4Opts_t,
    nvcompBatchedSnappyDefaultOpts,
    nvcompBatchedSnappyOpts_t,
    nvcompManagerBase,
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

    def __dealloc__(self):
        # `ManagedManager` uses a temporary object, self._mgr
        # to retain a reference count to the Manager created by
        # create_manager. If it is present, then the `shared_ptr`
        # system will free self._impl. Otherwise, we need to free
        # self._iNonempl
        if self._mgr == nullptr:
            del self._impl

    def configure_compression(self, decomp_buffer_size):
        cdef shared_ptr[CompressionConfig] partial = make_shared[
            CompressionConfig](
                self._impl.configure_compression(decomp_buffer_size)
        )
        self._compression_config = make_shared[CompressionConfig](
            (move(partial.get()[0]))
        )
        cdef const CompressionConfig* compression_config_ptr = \
            self._compression_config.get()
        return {
            "uncompressed_buffer_size": compression_config_ptr.
            uncompressed_buffer_size,
            "max_compressed_buffer_size": compression_config_ptr.
            max_compressed_buffer_size,
            "num_chunks": compression_config_ptr.num_chunks
        }

    def compress(self, Array decomp_buffer, Array comp_buffer):
        cdef uintptr_t comp_buffer_ptr = comp_buffer.ptr
        self._impl.compress(
            <const uint8_t*>decomp_buffer.ptr,
            <uint8_t*>comp_buffer_ptr,
            <CompressionConfig&>self._compression_config.get()[0]
        )
        size = self._impl.get_compressed_output_size(
            <uint8_t*>comp_buffer_ptr
        )
        return size

    def configure_decompression_with_compressed_buffer(
        self,
        Array comp_buffer
    ) -> dict:
        cdef shared_ptr[DecompressionConfig] partial = make_shared[
            DecompressionConfig](self._impl.configure_decompression(
                <uint8_t*>comp_buffer.ptr
            )
        )
        self._decompression_config = make_shared[DecompressionConfig](
            (move(partial.get()[0]))
        )
        cdef const DecompressionConfig* decompression_config_ptr = \
            self._decompression_config.get()
        return {
            "decomp_data_size": decompression_config_ptr.decomp_data_size,
            "num_chunks": decompression_config_ptr.num_chunks
        }

    def decompress(
        self,
        Array decomp_buffer,
        Array comp_buffer,
    ):
        self._impl.decompress(
            <uint8_t*>decomp_buffer.ptr,
            <const uint8_t*>comp_buffer.ptr,
            <DecompressionConfig&>self._decompression_config.get()[0]
        )

    def get_compressed_output_size(self, Array comp_buffer):
        return self._impl.get_compressed_output_size(
            <uint8_t*>comp_buffer.ptr
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
            <nvcompBatchedANSOpts_t>nvcompBatchedANSDefaultOpts,  # TODO
            <cudaStream_t><void*>0,  # TODO
            device_id
        )


cdef class _BitcompManager(_nvcompManager):
    def __cinit__(
        self,
        nvcompType_t data_type,
        int bitcomp_algo,
        user_stream,
        const int device_id
    ):
        self._impl = <nvcompManagerBase*>new BitcompManager(
            0, # TODO
            <nvcompBatchedBitcompFormatOpts>nvcompBatchedBitcompDefaultOpts,  # TODO
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
            0, # TODO
            <nvcompBatchedCascadedOpts_t>nvcompBatchedCascadedDefaultOpts,  # TODO
            <cudaStream_t><void*>0,  # TODO
            device_id,
        )


cdef class _GdeflateManager(_nvcompManager):
    def __cinit__(
        self,
        int chunk_size,
        int algo,
        user_stream,
        const int device_id
    ):
        self._impl = <nvcompManagerBase*>new GdeflateManager(
            chunk_size,
            <nvcompBatchedGdeflateOpts_t>nvcompBatchedGdeflateDefaultOpts,  # TODO
            <cudaStream_t><void*>0,  # TODO
            device_id
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
        # cdef cudaStream_t stream = <cudaStream_t><void*>user_stream
        self._impl = <nvcompManagerBase*>new LZ4Manager(
            uncomp_chunk_size,
            <nvcompBatchedLZ4Opts_t>nvcompBatchedLZ4DefaultOpts,
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
            <nvcompBatchedSnappyOpts_t>nvcompBatchedSnappyDefaultOpts,
            <cudaStream_t><void*>0,  # TODO
            device_id
        )


cdef class _ManagedManager(_nvcompManager):
    def __init__(self, compressed_buffer):
        cdef shared_ptr[nvcompManagerBase] _mgr = create_manager(
            <uint8_t*><uintptr_t>compressed_buffer.ptr
        )
        self._mgr = _mgr
        self._impl = move(_mgr).get()
