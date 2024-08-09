# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3


cdef extern from "<future>" namespace "std" nogil:
    cdef cppclass future[T]:
        future() except +
        T get() except +


cdef extern from "<kvikio/stream.hpp>" nogil:
    cdef cppclass cpp_StreamFuture "kvikio::StreamFuture":
        cpp_StreamFuture() except +
        size_t check_bytes_done() except +


cdef class IOFutureStream:
    cdef cpp_StreamFuture _handle

cdef IOFutureStream _wrap_stream_future(cpp_StreamFuture &fut)

cdef class IOFuture:
    cdef future[size_t] _handle

cdef IOFuture _wrap_io_future(future[size_t] &fut)
