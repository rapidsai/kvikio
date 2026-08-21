# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport int64_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string


cdef extern from "<kvikio/observation.hpp>" nogil:
    cdef cppclass cpp_SystemDuration "std::chrono::system_clock::duration":
        int64_t count()

    cdef cppclass cpp_SystemTimePoint "std::chrono::system_clock::time_point":
        cpp_SystemDuration time_since_epoch()

    cdef cppclass cpp_ClockAnchor "kvikio::ClockAnchor":
        cpp_SystemTimePoint to_wall_clock(cpp_TimePoint time)

    cdef cppclass cpp_ClockDuration "kvikio::Clock::duration":
        int64_t count()

    cdef cppclass cpp_Duration "kvikio::Duration":
        int64_t count()

    cdef cppclass cpp_TimePoint "kvikio::TimePoint":
        cpp_ClockDuration time_since_epoch()


cdef extern from "<chrono>" nogil:
    cpp_Duration to_ns "std::chrono::duration_cast<kvikio::Duration>"(
        cpp_ClockDuration duration
    )
    cpp_Duration system_to_ns "std::chrono::duration_cast<kvikio::Duration>"(
        cpp_SystemDuration duration
    )


cdef extern from "<kvikio/observation.hpp>" namespace "kvikio" nogil:
    const size_t num_io_backends


cdef extern from "<kvikio/statistics/summary.hpp>" nogil:
    cdef cppclass cpp_BackendTotals "kvikio::statistics::Summary::BackendTotals":
        uint64_t num_ops
        uint64_t bytes_transferred
        cpp_Duration total_duration
        uint64_t num_errors

    cdef cppclass cpp_Summary "kvikio::statistics::Summary":
        cpp_TimePoint start
        cpp_TimePoint end
        cpp_ClockAnchor anchor
        uint64_t num_ops
        uint64_t num_reads
        uint64_t num_writes
        uint64_t bytes_requested
        uint64_t bytes_transferred
        uint64_t bytes_read
        uint64_t bytes_written
        uint64_t num_errors
        cpp_Duration busy
        cpp_Duration total_duration
        cpp_BackendTotals[5] by_backend
        cpp_Duration wall() except +
        double busy_bytes_per_sec() except +
        double busy_fraction() except +
        cpp_Duration mean_duration() except +
        cpp_Summary since(const cpp_Summary& previous) except +
        string to_json() except +
        string report() except +

    cdef cppclass cpp_SummaryMonitor "kvikio::statistics::SummaryMonitor":
        cpp_SummaryMonitor() except +
        cpp_Summary get() except +
        void reset() except +
        cpp_Summary since(const cpp_Summary& previous) except +
        void stop() except +


cdef extern from *:
    """
    #include <cstddef>
    #include <cstring>
    #include <string>
    #include <vector>

    #include <kvikio/observation.hpp>
    #include <kvikio/statistics/summary.hpp>

    // `by_backend` below is declared with a literal length.
    static_assert(kvikio::num_io_backends == 5, "update `by_backend` in statistics.pyx");

    namespace {
    std::string kvikio_backend_name(std::size_t index)
    {
      return std::string{kvikio::to_string(static_cast<kvikio::IoBackend>(index))};
    }

    std::string kvikio_summary_to_bytes(kvikio::statistics::Summary const& summary)
    {
      auto const bytes = summary.serialize();
      return std::string{reinterpret_cast<char const*>(bytes.data()), bytes.size()};
    }

    kvikio::statistics::Summary kvikio_summary_from_bytes(std::string const& bytes)
    {
      std::vector<std::byte> buffer(bytes.size());
      if (!bytes.empty()) { std::memcpy(buffer.data(), bytes.data(), bytes.size()); }
      return kvikio::statistics::Summary::deserialize(buffer);
    }
    }  // namespace
    """
    string kvikio_backend_name(size_t index) except +
    string kvikio_summary_to_bytes(const cpp_Summary& summary) except +
    cpp_Summary kvikio_summary_from_bytes(const string& bytes) except +


# The names the C++ report prints, in the order of `kvikio::IoBackend`, which is what
# `by_backend` is indexed by.
_BACKEND_NAMES = tuple(kvikio_backend_name(i).decode() for i in range(num_io_backends))


cdef class Summary:
    """Wrapper of the C++ class kvikio::statistics::Summary"""

    cdef cpp_Summary _handle

    @staticmethod
    cdef Summary _from_cpp(cpp_Summary handle):
        cdef Summary ret = Summary.__new__(Summary)
        ret._handle = handle
        return ret

    def as_dict(self) -> dict:
        """Every field of `kvikio.Summary`, read in one pass."""
        return {
            "start_unix_ns": system_to_ns(
                self._handle.anchor.to_wall_clock(self._handle.start).time_since_epoch()
            ).count(),
            "end_unix_ns": system_to_ns(
                self._handle.anchor.to_wall_clock(self._handle.end).time_since_epoch()
            ).count(),
            "num_ops": self._handle.num_ops,
            "num_reads": self._handle.num_reads,
            "num_writes": self._handle.num_writes,
            "bytes_requested": self._handle.bytes_requested,
            "bytes_transferred": self._handle.bytes_transferred,
            "bytes_read": self._handle.bytes_read,
            "bytes_written": self._handle.bytes_written,
            "num_errors": self._handle.num_errors,
            "busy_ns": self._handle.busy.count(),
            "total_duration_ns": self._handle.total_duration.count(),
            "by_backend": {
                name: {
                    "num_ops": self._handle.by_backend[i].num_ops,
                    "bytes_transferred": self._handle.by_backend[i].bytes_transferred,
                    "total_duration_ns": self._handle.by_backend[i].total_duration.count(),
                    "num_errors": self._handle.by_backend[i].num_errors,
                }
                for i, name in enumerate(_BACKEND_NAMES)
            },
            "wall_ns": self._handle.wall().count(),
            "busy_bytes_per_sec": self._handle.busy_bytes_per_sec(),
            "busy_fraction": self._handle.busy_fraction(),
            "mean_duration_ns": self._handle.mean_duration().count(),
        }

    def since(self, Summary previous not None) -> Summary:
        return Summary._from_cpp(self._handle.since(previous._handle))

    def to_json(self) -> str:
        return self._handle.to_json().decode()

    def report(self) -> str:
        return self._handle.report().decode()

    def serialize(self) -> bytes:
        return kvikio_summary_to_bytes(self._handle)

    @staticmethod
    def deserialize(data: bytes) -> Summary:
        return Summary._from_cpp(kvikio_summary_from_bytes(data))


cdef class SummaryMonitor:
    """Wrapper of the C++ class kvikio::statistics::SummaryMonitor"""

    cdef unique_ptr[cpp_SummaryMonitor] _handle

    def __init__(self):
        self._handle = make_unique[cpp_SummaryMonitor]()

    def get(self) -> Summary:
        return Summary._from_cpp(self._handle.get().get())

    def reset(self) -> None:
        self._handle.get().reset()

    def since(self, Summary previous not None) -> Summary:
        return Summary._from_cpp(self._handle.get().since(previous._handle))

    def stop(self) -> None:
        self._handle.get().stop()
