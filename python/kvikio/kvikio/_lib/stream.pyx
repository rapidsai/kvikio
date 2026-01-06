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


def stream_register(stream: uintptr_t, flags: int) -> None:
    cdef CUstream cpp_stream = <CUstream>stream
    cdef unsigned int cpp_flags = flags
    with nogil:
        cpp_stream_register(cpp_stream, cpp_flags)


def stream_deregister(stream) -> None:
    cdef CUstream cpp_stream = <CUstream>stream
    with nogil:
        cpp_stream_deregister(cpp_stream)
