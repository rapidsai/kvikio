# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.utility cimport move


cdef extern from "<kvikio/utils.hpp>" namespace "kvikio" nogil:
    bool is_future_done[T](const T& future) except +


cdef class IOFutureStream:
    """Wrap a C++ StreamFuture in a Python object"""
    def check_bytes_done(self) -> int:
        cdef size_t bytes_done
        with nogil:
            bytes_done = self._handle.check_bytes_done()
        return bytes_done


cdef IOFutureStream _wrap_stream_future(cpp_StreamFuture &fut):
    """Wrap a C++ future (of a `size_t`) in a `IOFuture` instance"""
    ret = IOFutureStream()
    ret._handle = move(fut)
    return ret


cdef class IOFuture:
    """C++ future for CuFile reads and writes"""
    def get(self) -> int:
        cdef size_t ret
        with nogil:
            ret = self._handle.get()
        return ret

    def done(self) -> bool:
        cdef bool result
        with nogil:
            result = is_future_done(self._handle)
        return result


cdef IOFuture _wrap_io_future(future[size_t] &fut):
    """Wrap a C++ future (of a `size_t`) in a `IOFuture` instance"""
    ret = IOFuture()
    ret._handle = move(fut)
    return ret
