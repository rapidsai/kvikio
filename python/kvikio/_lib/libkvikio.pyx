# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import pathlib
from typing import Optional

from libc.stdint cimport uintptr_t
from libcpp.utility cimport move, pair

from . cimport kvikio_cxx_api
from .arr cimport Array
from .kvikio_cxx_api cimport CUstream, FileHandle, StreamFuture, future, is_future_done


cdef class IOFutureStream:
    """Wrap a C++ StreamFuture in a Python object"""
    cdef StreamFuture _handle

    def check_bytes_done(self) -> int:
        return self._handle.check_bytes_done()


cdef IOFutureStream _wrap_stream_future(StreamFuture &fut):
    """Wrap a C++ future (of a `size_t`) in a `IOFuture` instance"""
    ret = IOFutureStream()
    ret._handle = move(fut)
    return ret


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


cdef IOFuture _wrap_io_future(future[size_t] &fut):
    """Wrap a C++ future (of a `size_t`) in a `IOFuture` instance"""
    ret = IOFuture()
    ret._handle = move(fut)
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


def compat_mode() -> bool:
    return kvikio_cxx_api.compat_mode()


def compat_mode_reset(enable: bool) -> None:
    kvikio_cxx_api.compat_mode_reset(enable)


def thread_pool_nthreads() -> int:
    return kvikio_cxx_api.thread_pool_nthreads()


def thread_pool_nthreads_reset(nthreads: int) -> None:
    kvikio_cxx_api.thread_pool_nthreads_reset(nthreads)


def task_size() -> int:
    return kvikio_cxx_api.task_size()


def task_size_reset(nbytes: int) -> None:
    kvikio_cxx_api.task_size_reset(nbytes)


def gds_threshold() -> int:
    return kvikio_cxx_api.gds_threshold()


def gds_threshold_reset(nbytes: int) -> None:
    kvikio_cxx_api.gds_threshold_reset(nbytes)


cdef pair[uintptr_t, size_t] _parse_buffer(buf, size, bint accept_host_buffer) except *:
    """Parse `buf` and `size` argument and return a pointer and nbytes"""
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    if not arr._contiguous():
        raise ValueError("Array must be contiguous")
    if not accept_host_buffer and not arr.cuda:
        raise ValueError("Non-CUDA buffers not supported")
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

    def pread(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, True)
        return _wrap_io_future(
            self._handle.pread(
                <void*>info.first,
                info.second,
                file_offset,
                task_size if task_size else kvikio_cxx_api.task_size()
            )
        )

    def pwrite(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, True)
        return _wrap_io_future(
            self._handle.pwrite(
                <void*>info.first,
                info.second,
                file_offset,
                task_size if task_size else kvikio_cxx_api.task_size()
            )
        )

    def read(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return self._handle.read(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def write(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return self._handle.write(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def read_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                   st: uintptr_t) -> IOFutureStream:
        stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return _wrap_stream_future(self._handle.read_async(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
            stream,
        ))

    def write_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                    st: uintptr_t) -> IOFutureStream:
        stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return _wrap_stream_future(self._handle.write_async(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
            stream,
        ))


cdef class DriverProperties:
    cdef kvikio_cxx_api.DriverProperties _handle

    @property
    def is_gds_available(self) -> bool:
        try:
            return self._handle.is_gds_available()
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
