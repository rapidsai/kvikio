/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Internal support code for the KvikIO Nsight Systems NIC plugin. It reads network byte counters
// from sysfs and converts them to rates. It is compiled directly into the plugin and its test. It
// is not part of libkvikio and is not an installed public header.

namespace kvikio::nsys_plugin {

namespace constants {
inline constexpr double bytes_per_mib  = 1024.0 * 1024.0;
inline constexpr char const* sysfs_net = "/sys/class/net";
}  // namespace constants

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
 * @brief One bandwidth sample for a single interface: the receive and transmit rates.
 *
 * The field names are deliberately unit-neutral. `compute_rates` produces the values in MiB/s, but
 * the unit is not baked into the type; a consumer attaches it separately (for example as NVTX
 * counter semantics), so switching units later does not ripple into these names.
 */
struct NicRates {
  double rx;  ///< Receive rate.
  double tx;  ///< Transmit rate.
};

/**
 * @brief Check whether a network interface is up.
 *
 * @param name Interface name.
 * @return True only when operstate is "up" or "unknown". False for any other state or if it cannot
 * be read.
 */
[[nodiscard]] bool iface_is_up(std::string const& name);

/**
 * @brief Enumerate the default set of interfaces to monitor.
 *
 * @return All "up" non-loopback interfaces under `/sys/class/net`, sorted for a stable order.
 */
[[nodiscard]] std::vector<std::string> default_interfaces();

/**
 * @brief Reads the byte counters of a fixed set of network interfaces.
 *
 * The two sysfs paths per interface are precomputed once at construction, so each `read()` performs
 * only the 2N pseudo-file reads, with no directory iteration and no path construction. This keeps
 * the per-tick footprint of the sampling loop minimal and independent of how many other interfaces
 * exist on the host.
 */
class NicCounterReader {
 public:
  /**
   * @brief Construct a reader over the given interfaces.
   *
   * @param interfaces Interface names to read, in the order `read()` reports them.
   */
  explicit NicCounterReader(std::vector<std::string> interfaces);

  /**
   * @brief Read the current cumulative byte counters of every configured interface.
   *
   * @return One entry per interface, in constructor order. An entry is std::nullopt when either of
   * its counters is unreadable (for example the interface was removed).
   */
  [[nodiscard]] std::vector<std::optional<NicCounters>> read() const;

  /**
   * @brief The interfaces this reader was constructed with.
   */
  [[nodiscard]] std::vector<std::string> const& interfaces() const noexcept;

 private:
  std::vector<std::string> _interfaces;
  // Precomputed `/sys/class/net/<iface>/statistics/{rx,tx}_bytes` paths, parallel to `_interfaces`.
  std::vector<std::string> _rx_paths;
  std::vector<std::string> _tx_paths;
};

/**
 * @brief Convert a pair of cumulative counter samples into receive and transmit rates in MiB/s.
 *
 * @param prev The earlier counter sample.
 * @param cur The later counter sample.
 * @param dt_seconds Elapsed wall-clock seconds between the two samples.
 * @return Rates in MiB/s. A direction whose counter did not advance yields zero for that direction
 * (this guards the first sample as well as a counter wrap or reset). Both rates are zero when @p
 * dt_seconds is not positive.
 */
[[nodiscard]] NicRates compute_rates(NicCounters const& prev,
                                     NicCounters const& cur,
                                     double dt_seconds) noexcept;

}  // namespace kvikio::nsys_plugin
