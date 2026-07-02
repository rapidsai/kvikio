/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nic_monitor.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

namespace kvikio::experimental {

namespace {

/**
 * @brief Read a sysfs pseudo-file in a single read().
 *
 * @param path Path of the pseudo-file to read.
 * @param buf Scratch buffer that backs the returned view. At most `buf.size()` bytes are read, so
 * it never overflows.
 * @return A view of the bytes read (into @p buf), or std::nullopt if the file cannot be read. The
 * string view is not necessarily null terminated.
 */
[[nodiscard]] std::optional<std::string_view> read_pseudo_file(char const* path,
                                                               std::span<char> buf) noexcept
{
  auto const fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { return std::nullopt; }
  auto const n = ::read(fd, buf.data(), buf.size());
  ::close(fd);
  if (n <= 0) { return std::nullopt; }
  return std::string_view{buf.data(), static_cast<std::size_t>(n)};
}

/**
 * @brief Parse a single unsigned integer from a sysfs pseudo-file.
 *
 * @param path Path of the pseudo-file to read.
 * @return The parsed value, or std::nullopt if the file is unparseable.
 */
[[nodiscard]] std::optional<std::uint64_t> read_u64_file(std::string const& path) noexcept
{
  // A uint64 counter is at most 20 digits, so 32 bytes is ample (20 digits + newline)
  std::array<char, 32> buf;
  auto const content = read_pseudo_file(path.c_str(), buf);
  if (!content.has_value()) { return std::nullopt; }
  std::uint64_t value{};
  // from_chars is locale-independent, non-allocating, and non-throwing, and stops at the trailing
  // newline.
  [[maybe_unused]] auto const [_, ec] =
    std::from_chars(content->data(), content->data() + content->size(), value);
  if (ec != std::errc{}) { return std::nullopt; }
  return value;
}

}  // namespace

bool iface_is_up(std::string const& name)
{
  auto const path = std::string{constants::sysfs_net} + "/" + name + "/operstate";
  std::array<char, 16> buf;
  auto const content = read_pseudo_file(path.c_str(), buf);
  if (!content.has_value()) { return false; }
  std::string_view state = content.value();
  while (!state.empty() && (state.back() == '\n' || state.back() == ' ')) {
    state.remove_suffix(1);
  }
  // Some working NICs report "unknown" (loopback or virtual NICs), so this state should be
  // accepted. The other states (down, notpresent, lowerlayerdown, testing, dormant) should be
  // excluded.
  return state == "up" || state == "unknown";
}

std::vector<std::string> default_interfaces()
{
  std::vector<std::string> result;
  std::error_code ec;
  std::filesystem::directory_iterator it{std::filesystem::path{constants::sysfs_net}, ec};
  if (ec) { return result; }
  for (auto const& entry : it) {
    auto name = entry.path().filename().string();
    if (name == "lo") { continue; }
    if (!iface_is_up(name)) { continue; }
    result.push_back(std::move(name));
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::map<std::string, NicCounters> read_nic_counters()
{
  std::map<std::string, NicCounters> result;
  std::error_code ec;
  std::filesystem::directory_iterator it{std::filesystem::path{constants::sysfs_net}, ec};
  if (ec) { return result; }
  for (auto const& entry : it) {
    auto const name  = entry.path().filename().string();
    auto const stats = entry.path() / "statistics";
    auto const rx    = read_u64_file((stats / "rx_bytes").string());
    auto const tx    = read_u64_file((stats / "tx_bytes").string());
    // If an interface exposes no parseable byte counters, skip it.
    if (!rx.has_value() || !tx.has_value()) { continue; }
    result.emplace(name, NicCounters{rx.value(), tx.value()});
  }
  return result;
}

NicRates compute_rates(NicCounters const& prev, NicCounters const& cur, double dt_seconds) noexcept
{
  NicRates rates{0.0, 0.0};
  if (dt_seconds <= 0.0) { return rates; }
  if (cur.rx_bytes >= prev.rx_bytes) {
    auto const delta = cur.rx_bytes - prev.rx_bytes;
    rates.rx         = static_cast<double>(delta) / dt_seconds / constants::bytes_per_mib;
  }
  if (cur.tx_bytes >= prev.tx_bytes) {
    auto const delta = cur.tx_bytes - prev.tx_bytes;
    rates.tx         = static_cast<double>(delta) / dt_seconds / constants::bytes_per_mib;
  }
  return rates;
}

}  // namespace kvikio::experimental
