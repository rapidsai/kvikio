# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import os
import pathlib
from typing import Tuple

from libc.stdint cimport uint32_t, uintptr_t
from libcpp.utility cimport move, pair

from . cimport kvikio_cxx_api
from .arr cimport Array
from .kvikio_cxx_api cimport FileHandle, future, is_future_done


cdef class IOFuture:
    """C++ future for CuFile reads and writes"""
    cdef future[size_t] _handle

    def get(self) -> int:
        cdef size_t ret
        with nogil:
            ret = self._handle.get()
        return ret

    def done(self) -> bool:
        return is_future_done(self._handle)


cdef IOFuture _wrap_io_future(future[size_t] &future):
    """Wrap a C++ future (of a `size_t`) in a `IOFuture` instance"""
    ret = IOFuture()
    ret._handle = move(future)
    return ret


def memory_register(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    kvikio_cxx_api.memory_register(<void*>arr.ptr)


def memory_deregister(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    kvikio_cxx_api.memory_deregister(<void*>arr.ptr)


def compat_mode() -> int:
    return kvikio_cxx_api.compat_mode()


def compat_mode_reset(enable: bool) -> None:
    kvikio_cxx_api.compat_mode_reset(enable)


def thread_pool_nthreads() -> int:
    return kvikio_cxx_api.thread_pool_nthreads()


def thread_pool_nthreads_reset(nthreads: int) -> None:
    kvikio_cxx_api.thread_pool_nthreads_reset(nthreads)


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

    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        return self._handle.fd()

    def open_flags(self) -> int:
        return self._handle.fd_open_flags()

    def pread(self, buf, size: int, file_offset: int, ntasks) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size)
        return _wrap_io_future(
            self._handle.pread(
                <void*>info.first,
                info.second,
                file_offset,
                ntasks if ntasks else kvikio_cxx_api.thread_pool_nthreads()
            )
        )

    def pwrite(self, buf, size: int, file_offset: int, ntasks) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size)
        return _wrap_io_future(
            self._handle.pwrite(
                <void*>info.first,
                info.second,
                file_offset,
                ntasks if ntasks else kvikio_cxx_api.thread_pool_nthreads()
            )
        )

    def read(self, buf, size: int, file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size)
        return self._handle.read(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def write(self, buf, size: int, file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size)
        return self._handle.write(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )


cdef class DriverProperties:
    cdef kvikio_cxx_api.DriverProperties _handle

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
