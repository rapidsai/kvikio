# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import os
import pathlib
from typing import Tuple

cimport cufile_cxx_api
from cufile_cxx_api cimport FileHandle, future
from libc.stdint cimport uint32_t, uintptr_t
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

from arr cimport Array


def memory_register(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    cufile_cxx_api.memory_register(<void*>arr.ptr)

def memory_deregister(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    cufile_cxx_api.memory_deregister(<void*>arr.ptr)

def set_num_threads(nthread: int) -> None:
    cufile_cxx_api.reset(nthread)

def get_num_threads() -> int:
    return cufile_cxx_api.nthreads()

cdef pair[uintptr_t, size_t] _parse_buffer(buf, size):
    """Parse `buf` and `size` argument and return a pointer and nbytes"""
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    if not arr._contiguous():
        raise ValueError("Array must be C or F contiguous")
    if not arr.cuda:
        raise NotImplementedError("Non-CUDA buffers not implemented")
    cdef size_t nbytes
    if size is None:
        nbytes = arr.nbytes
    elif size > arr.nbytes:
        raise ValueError("Size is greater than the size of the buffer")
    else:
        nbytes = size
    return pair[uintptr_t, size_t](arr.ptr, nbytes)


cdef class CuFile:
    """ File handle for GPUDirect Storage (GDS) """
    cdef FileHandle _handle

    def __init__(self, file_path, flags="r"):
        self._handle = move(
            FileHandle(
                str.encode(str(pathlib.Path(file_path))),
                str.encode(str(flags))
            )
        )

    def close(self) -> None:
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        return self._handle.fd()

    def open_flags(self) -> int:
        return self._handle.fd_open_flags()

    def __enter__(self) -> CuFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read(self,
        buf, size: int = None, file_offset: int = 0, nthreads = None
    ) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size)
        return move(
            self._handle.pread(
                <void*>info.first,
                info.second,
                file_offset,
                nthreads if nthreads else get_num_threads()
            )
        ).get()


    def write(self,
        buf, size: int = None, file_offset: int = 0, nthreads = None
    ) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size)
        return move(
            self._handle.pwrite(
                <void*>info.first,
                info.second,
                file_offset,
                nthreads if nthreads else get_num_threads()
            )
        ).get()


cdef class DriverProperties:
    cdef cufile_cxx_api.DriverProperties _handle

    @property
    def is_gds_availabe(self) -> bool:
        try:
            return self._handle.is_gds_availabe()
        except RuntimeError:
            return False

    @property
    def major_version(self) -> bool:
        return self._handle.get_nvfs_major_version()

    @property
    def minor_version(self) -> bool:
        return self._handle.get_nvfs_minor_version()

    @property
    def allow_compat_mode(self) -> bool:
        return self._handle.get_nvfs_allow_compat_mode()

    @property
    def poll_mode(self) -> bool:
        return self._handle.get_nvfs_poll_mode()

    @poll_mode.setter
    def poll_mode(self, enable: bool) -> None:
        self._handle.set_nvfs_poll_mode(enable)

    @property
    def poll_thresh_size(self) -> int:
        return self._handle.get_nvfs_poll_thresh_size()

    @poll_thresh_size.setter
    def poll_thresh_size(self, size_in_kb: int) -> None:
        self._handle.set_nvfs_poll_thresh_size(size_in_kb)

    @property
    def max_device_cache_size(self) -> int:
        return self._handle.get_max_device_cache_size()

    @max_device_cache_size.setter
    def max_device_cache_size(self, size_in_kb: int) -> None:
        self._handle.set_max_device_cache_size(size_in_kb)

    @property
    def per_buffer_cache_size(self) -> int:
        return self._handle.get_per_buffer_cache_size()

    @property
    def max_pinned_memory_size(self) -> int:
        return self._handle.get_max_pinned_memory_size()

    @max_pinned_memory_size.setter
    def max_pinned_memory_size(self, size_in_kb: int) -> None:
        self._handle.set_max_pinned_memory_size(size_in_kb)

cdef class NVML:
    cdef cufile_cxx_api.NVML _handle

    def get_name(self) -> str:
        return self._handle.get_name().decode()

    def get_memory(self) -> Tuple[int, int]:
        cdef pair[size_t, size_t] info = self._handle.get_memory()
        return info.first, info.second

    def get_bar1_memory(self) -> Tuple[int, int]:
        cdef pair[size_t, size_t] info = self._handle.get_bar1_memory()
        return info.first, info.second
