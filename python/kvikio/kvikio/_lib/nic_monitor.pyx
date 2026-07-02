# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector


cdef extern from "<kvikio/experimental/nic_monitor.hpp>" \
        namespace "kvikio::experimental" nogil:
    cdef cppclass cpp_NicBandwidthMonitor "kvikio::experimental::NicBandwidthMonitor":
        cpp_NicBandwidthMonitor(double freq_hz, vector[string] interfaces) except +
        void start() except +
        void stop() except +
        bool running()
        const vector[string]& interfaces()


cdef class NicBandwidthMonitor:
    """Samples NIC receive bandwidth and emits it as NVTX counters.

    A background C++ thread differences the kernel byte counters at a fixed
    frequency and emits one NVTX float64 counter per interface (named
    ``nic_rx_MiBps.<iface>``) plus a summed ``nic_rx_MiBps.total``. The sampling
    runs entirely in C++ and never holds the Python GIL.

    Parameters
    ----------
    freq_hz
        Sampling frequency in hertz. Must be positive.
    interfaces
        Interface names to monitor. If ``None`` or empty, all UP non-loopback
        interfaces are selected when :meth:`start` is called.
    """
    cdef unique_ptr[cpp_NicBandwidthMonitor] _handle

    def __init__(self, double freq_hz=20.0, interfaces=None):
        cdef vector[string] cpp_ifaces
        if interfaces is not None:
            for iface in interfaces:
                cpp_ifaces.push_back(str(iface).encode())
        with nogil:
            self._handle = make_unique[cpp_NicBandwidthMonitor](
                freq_hz, move(cpp_ifaces)
            )

    def start(self) -> None:
        """Register the NVTX counters and launch the sampling thread."""
        with nogil:
            deref(self._handle).start()

    def stop(self) -> None:
        """Stop and join the sampling thread. Safe to call more than once."""
        with nogil:
            deref(self._handle).stop()

    def running(self) -> bool:
        """Whether the sampling thread is currently running."""
        cdef bool result
        with nogil:
            result = deref(self._handle).running()
        return result

    def interfaces(self) -> list:
        """The interfaces being monitored (populated after :meth:`start`)."""
        cdef vector[string] result
        with nogil:
            result = deref(self._handle).interfaces()
        return [iface.decode() for iface in result]

    def __enter__(self) -> "NicBandwidthMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
