/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

#include <kvikio/observation.hpp>

namespace kvikio::detail {

/// @brief Control whether a zero fractional part is omitted when formatting values.
enum class TrimZeroFraction : std::uint8_t {
  NO,  ///< Always keep the fractional part.
  YES  ///< Omit the fractional part when it consists only of zeros.
};

/**
 * @brief Format a byte count as a human-readable string using IEC units.
 *
 * Converts a byte count into a scaled string representation using binary (base-1024) units such
 * as KiB, MiB and GiB.
 *
 * Negative values are supported and are formatted with a leading minus sign, which is useful when
 * representing signed byte deltas.
 *
 * Examples:
 *   - 1024 bytes with 2 decimals gives `"1 KiB"`, or `"1.00 KiB"` when the fraction is kept.
 *   - 1536 bytes with 2 decimals gives `"1.50 KiB"`.
 *
 * @param nbytes Number of bytes. Unsigned, since every byte count KvikIO reports is a total that
 * only grows, and a signed parameter would render one past `INT64_MAX` as a negative.
 * @param num_decimals Number of decimal places to include.
 * @param trim_zero_fraction Whether to omit the fractional part when it consists only of zeros.
 * @return The formatted byte count.
 */
[[nodiscard]] std::string format_nbytes(
  std::uint64_t nbytes,
  int num_decimals                    = 2,
  TrimZeroFraction trim_zero_fraction = TrimZeroFraction::YES);

/**
 * @brief Format a byte rate as a human-readable string using SI units.
 *
 * Scales with decimal (base-1000) units, `kB/s`, `MB/s` and so on, as storage and network
 * throughput is quoted, unlike the binary units `format_nbytes()` uses for sizes.
 *
 * @param bytes_per_sec The rate.
 * @param num_decimals Number of decimal places to include.
 * @param trim_zero_fraction Whether to omit the fractional part when it consists only of zeros.
 * @return The formatted rate.
 */
[[nodiscard]] std::string format_rate(double bytes_per_sec,
                                      int num_decimals                    = 2,
                                      TrimZeroFraction trim_zero_fraction = TrimZeroFraction::YES);

/**
 * @brief Format a duration as a human-readable string.
 *
 * Scales to `ns`, `us`, `ms`, `s`, `min`, `h` or `d`, whichever keeps the value readable.
 *
 * Negative values are supported and are formatted with a leading minus sign, which is useful when
 * representing signed time deltas.
 *
 * @param duration The duration.
 * @param num_decimals Number of decimal places to include.
 * @param trim_zero_fraction Whether to omit the fractional part when it consists only of zeros.
 * @return The formatted duration.
 */
[[nodiscard]] std::string format_duration(
  Duration duration,
  int num_decimals                    = 2,
  TrimZeroFraction trim_zero_fraction = TrimZeroFraction::YES);

}  // namespace kvikio::detail
