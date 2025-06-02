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


cdef extern from "<kvikio/mmap.hpp>" namespace "kvikio" nogil:
    cdef cppclass CppMmapHandle "kvikio::MmapHandle":
        CppMmapHandle() noexcept
        CppMmapHandle(string file_path,
                      string flags,
                      size_t initial_size,
                      size_t initial_file_offset,
                      void* external_buf,
                      fcntl.mode_t mode) except +
        pair[void*, size_t] read(size_t size,
                                 size_t file_offset,
                                 bool prefault) except +

cdef class MmapHandle:
    cdef CppMmapHandle _handle

    def __init__(self,
                 file_path,
                 flags="r",
                 initial_size=0,
                 initial_file_offset=0,
                 external_buf=None,
                 mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH):
        path_bytes = str(pathlib.Path(file_path)).encode()
        flags_bytes = str(flags).encode()
        if external_buf is None:
            self._handle = move(CppMmapHandle(path_bytes,
                                              flags_bytes,
                                              initial_size,
                                              initial_file_offset,
                                              NULL,
                                              mode))
        else:
            self._external_buf_non_null(path_bytes,
                                        flags_bytes,
                                        initial_size,
                                        initial_file_offset,
                                        external_buf,
                                        mode)

    def _external_buf_non_null(self,
                               path_bytes,
                               flags_bytes,
                               initial_size,
                               initial_file_offset,
                               external_buf,
                               mode):
        """This function is used to work around a limitation where cdef cannot be
        placed in an if-else construct.
        """
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(external_buf,
                                                                  initial_size,
                                                                  True)
        self._handle = move(CppMmapHandle(path_bytes,
                                          flags_bytes,
                                          info.second,
                                          initial_file_offset,
                                          <void*>info.first,
                                          mode))

    # def read(self, size, file_offset=0, prefault=False):
    #     cdef pair[void*, size_t]
    #     std::pair<void*, std::size_t> read(std::size_t size,
    #                                         std::size_t file_offset = 0,
    #                                         bool prefault           = false);