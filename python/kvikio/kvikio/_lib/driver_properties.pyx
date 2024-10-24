# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3


from libcpp cimport bool


cdef extern from "<kvikio/cufile/driver.hpp>" nogil:
    cdef cppclass cpp_DriverProperties "kvikio::DriverProperties":
        cpp_DriverProperties() except +
        bool is_gds_available() except +
        unsigned int get_nvfs_major_version() except +
        unsigned int get_nvfs_minor_version() except +
        bool get_nvfs_allow_compat_mode() except +
        bool get_nvfs_poll_mode() except +
        size_t get_nvfs_poll_thresh_size() except +
        void set_nvfs_poll_mode(bool enable) except +
        void set_nvfs_poll_thresh_size(size_t size_in_kb) except +
        size_t get_max_device_cache_size() except +
        void set_max_device_cache_size(size_t size_in_kb) except +
        size_t get_per_buffer_cache_size() except +
        size_t get_max_pinned_memory_size() except +
        void set_max_pinned_memory_size(size_t size_in_kb) except +


cdef class DriverProperties:
    cdef cpp_DriverProperties _handle

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
