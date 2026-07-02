/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <map>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace kvikio::experimental {

/**
 * @brief Cumulative byte counters for a single network interface.
 *
 * The values are the kernel's running totals since boot, as exposed by
 * `/sys/class/net/<iface>/statistics/{rx,tx}_bytes`.
 */
struct NicCounters {
  std::uint64_t rx_bytes;  ///< Total bytes received since boot.
  std::uint64_t tx_bytes;  ///< Total bytes transmitted since boot.
};

/**
 * @brief Read the current cumulative byte counters for every network interface.
 *
 * Reads `/sys/class/net/<iface>/statistics/{rx,tx}_bytes`. Interfaces whose counters cannot be read
 * are skipped rather than reported as an error.
 *
 * @return A map from interface name to its cumulative counters. Empty if `/sys/class/net` is
 * unavailable.
 */
std::map<std::string, NicCounters> read_nic_counters();

/**
 * @brief Samples NIC bandwidth and emits it as NVTX counter groups.
 *
 * On `start()`, a background `std::jthread` differences the kernel byte counters at a fixed
 * frequency and emits one NVTX counter group per monitored interface (named `nic_MiBps.<iface>`)
 * plus a summed `nic_MiBps.total`, in the `libkvikio` NVTX domain. Each group carries the receive
 * and transmit rates (`rx_MiBps`, `tx_MiBps`) sampled together, so they render as a single grouped
 * counter track on the Nsight Systems timeline.
 */
class NicBandwidthMonitor {
 public:
  /**
   * @brief Construct a monitor.
   *
   * @param freq_hz Sampling frequency in hertz. Must be positive.
   * @param interfaces Interfaces to monitor. If empty, all UP non-loopback interfaces are selected
   * once at `start()`.
   *
   * @exception std::invalid_argument if @p freq_hz is not positive.
   */
  explicit NicBandwidthMonitor(double freq_hz = 20.0, std::vector<std::string> interfaces = {});

  /**
   * @brief Stop the sampling thread if it is running.
   */
  ~NicBandwidthMonitor();

  // Non-copyable
  NicBandwidthMonitor(NicBandwidthMonitor const&)            = delete;
  NicBandwidthMonitor& operator=(NicBandwidthMonitor const&) = delete;

  /**
   * @brief Select interfaces (if not given), register NVTX counters, and launch the sampling
   * thread. Has no effect if already running.
   */
  void start();

  /**
   * @brief Signal the sampling thread to stop and join it. Has no effect if not running.
   */
  void stop();

  /**
   * @brief Whether the sampling thread is currently running.
   */
  [[nodiscard]] bool running() const noexcept;

  /**
   * @brief The interfaces being monitored (populated at `start()`).
   */
  [[nodiscard]] std::vector<std::string> const& interfaces() const noexcept;

 private:
  void run(std::stop_token stop_token);

  double _freq_hz;
  std::vector<std::string> _interfaces;

  // NVTX counter ids, parallel to `_interfaces`, plus the total counter id.
  std::vector<std::uint64_t> _counter_ids;
  std::uint64_t _total_counter_id{0};

  // Declared after the members used by `run()`, so the jthread's auto-join on destruction happens
  // before those members are destroyed.
  std::jthread _thread;

  bool _running{false};
};

}  // namespace kvikio::experimental
