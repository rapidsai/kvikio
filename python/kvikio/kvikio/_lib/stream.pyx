# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uintptr_t


cdef extern from "cuda.h":
    ctypedef void* CUstream


cdef extern from "<kvikio/stream.hpp>" nogil:
    void cpp_stream_register "kvikio::stream_register"(CUstream stream, unsigned flags) except +
    void cpp_stream_deregister "kvikio::stream_deregister"(CUstream stream) except +


# Here stream's type can be annotated in two ways:
#
# - Case 1: stream: int
#   `stream` is a Python object (int is always a Python type hint).
#   Requires two-level casting to extract the integer value first:
#       cdef CUstream cpp_stream = <CUstream><uintptr_t>stream
#
# - Case 2: stream: uintptr_t
#   `stream` is a C uintptr_t (because uintptr_t is cimported from libc.stdint).
#   Cython auto-converts the Python int to C at function entry, so only one cast needed:
#       cdef CUstream cpp_stream = <CUstream>stream
#
# Note: In Case 1, using a single-level cast `<CUstream>stream` will compile, but
# it casts the Python object's memory address rather than extracting the integer
# value, resulting in a wrong pointer being passed silently.
def stream_register(stream: uintptr_t, flags: int) -> None:
    cdef CUstream cpp_stream = <CUstream>stream
    cdef unsigned int cpp_flags = flags
    with nogil:
        cpp_stream_register(cpp_stream, cpp_flags)


def stream_deregister(stream: uintptr_t) -> None:
    cdef CUstream cpp_stream = <CUstream>stream
    with nogil:
        cpp_stream_deregister(cpp_stream)
