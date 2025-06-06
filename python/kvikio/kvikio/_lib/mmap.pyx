# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import pathlib

from posix cimport fcntl, stat

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move, pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/mmap.hpp>" namespace "kvikio" nogil:
    cdef cppclass CppMmapHandle "kvikio::MmapHandle":
        CppMmapHandle() noexcept
        CppMmapHandle(string file_path, string flags, size_t initial_size,
                      size_t initial_file_offset, fcntl.mode_t mode) except +
        size_t initial_size()
        size_t initial_file_offset()
        size_t file_size() except +
        void close()
        bool closed()
        size_t read(void* buf, size_t size, size_t file_offset) except +
        future[size_t] pread(void* buf, size_t size, size_t file_offset,
                             size_t task_size) except +

cdef class MmapHandle:
    cdef CppMmapHandle _handle

    def __init__(self, file_path: os.PathLike,
                 flags: str = "r",
                 initial_size: int = 0,
                 initial_file_offset: int = 0,
                 mode: int = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH):
        path_bytes = str(pathlib.Path(file_path)).encode()
        flags_bytes = str(flags).encode()
        self._handle = move(CppMmapHandle(path_bytes,
                                          flags_bytes,
                                          initial_size,
                                          initial_file_offset,
                                          mode))

    def initial_size() -> int:
        return self._handle.initial_size()

    def initial_file_offset() -> int:
        return self._handle.initial_file_offset()

    def file_size() -> int:
        return self._handle.file_size()

    def close(self) -> None:
        self._handle.close()

    def closed(self) -> bool:
        return self._handle.closed()

    def read(self, buf, size: int, file_offset: int = 0) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return self._handle.read(<void*>info.first,
                                 info.second,
                                 file_offset)

    def pread(self, buf, size: int, file_offset: int = 0,
              task_size: int = 0) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(self._handle.pread(<void*>info.first,
                               info.second,
                               file_offset,
                               task_size))
