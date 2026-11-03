/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include <kvikio/detail/string_utils.hpp>

namespace kvikio::detail {

namespace {

/// Drop the fractional part of `"1.000"` and leave `"1.500"` alone.
std::string trim_zero_fraction(std::string value)
{
  auto const point = value.find('.');
  if (point == std::string::npos) { return value; }
  if (value.find_first_not_of('0', point + 1) != std::string::npos) { return value; }
  value.erase(point);
  return value;
}

/// Format the magnitude, sign it, and append the unit.
std::string format(
  double value, double magnitude, char const* unit, int num_decimals, TrimZeroFraction trim)
{
  std::ostringstream os;
  if (value < 0) { os << '-'; }
  os << std::fixed << std::setprecision(num_decimals) << magnitude;

  auto ret = os.str();
  if (trim == TrimZeroFraction::YES && num_decimals > 0) { ret = trim_zero_fraction(ret); }
  ret += ' ';
  ret += unit;
  return ret;
}

}  // namespace

std::string format_nbytes(std::uint64_t nbytes,
                          int num_decimals,
                          TrimZeroFraction trim_zero_fraction)
{
  constexpr std::array units{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

  auto magnitude       = static_cast<double>(nbytes);
  std::size_t unit_idx = 0;
  while (magnitude >= 1024.0 && unit_idx + 1 < units.size()) {
    magnitude /= 1024.0;
    ++unit_idx;
  }
  return format(
    static_cast<double>(nbytes), magnitude, units[unit_idx], num_decimals, trim_zero_fraction);
}

std::string format_rate(double bytes_per_sec, int num_decimals, TrimZeroFraction trim_zero_fraction)
{
  constexpr std::array units{"B/s", "kB/s", "MB/s", "GB/s", "TB/s", "PB/s"};

  auto magnitude       = std::abs(bytes_per_sec);
  std::size_t unit_idx = 0;
  if (std::isfinite(magnitude)) {
    while (magnitude >= 1000.0 && unit_idx + 1 < units.size()) {
      magnitude /= 1000.0;
      ++unit_idx;
    }
  }
  return format(bytes_per_sec, magnitude, units[unit_idx], num_decimals, trim_zero_fraction);
}

std::string format_duration(Duration duration,
                            int num_decimals,
                            TrimZeroFraction trim_zero_fraction)
{
  struct Unit {
    char const* name;
    double scale;
  };
  // Largest first, so the first that the value reaches is the one to use.
  constexpr std::array large_units{Unit{"d", 86400.0}, Unit{"h", 3600.0}, Unit{"min", 60.0}};
  constexpr std::array small_units{
    Unit{"s", 1.0}, Unit{"ms", 1e-3}, Unit{"us", 1e-6}, Unit{"ns", 1e-9}};

  auto const seconds = std::chrono::duration<double>{duration}.count();
  auto magnitude     = std::abs(seconds);
  char const* unit   = "s";

  if (std::isfinite(magnitude)) {
    for (auto const& candidate : large_units) {
      if (magnitude >= candidate.scale) {
        magnitude /= candidate.scale;
        unit = candidate.name;
        break;
      }
    }
    if (magnitude < 1.0) {
      for (auto const& candidate : small_units) {
        if (magnitude >= candidate.scale) {
          magnitude /= candidate.scale;
          unit = candidate.name;
          break;
        }
      }
    }
  }
  return format(seconds, magnitude, unit, num_decimals, trim_zero_fraction);
}

}  // namespace kvikio::detail
