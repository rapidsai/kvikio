/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>

#include <gtest/gtest.h>

#include <kvikio/detail/string_utils.hpp>

using kvikio::detail::format_duration;
using kvikio::detail::format_nbytes;
using kvikio::detail::format_rate;
using kvikio::detail::TrimZeroFraction;

TEST(StringUtilsTest, byte_counts_scale_by_1024)
{
  EXPECT_EQ(format_nbytes(0), "0 B");
  EXPECT_EQ(format_nbytes(1023), "1023 B");
  EXPECT_EQ(format_nbytes(1024), "1 KiB") << "the fraction is trimmed when it is all zeros";
  EXPECT_EQ(format_nbytes(1536), "1.50 KiB");
  EXPECT_EQ(format_nbytes(1024 * 1024), "1 MiB");
  // Totals only grow, so the parameter is unsigned and one past `INT64_MAX` is not a negative.
  EXPECT_EQ(format_nbytes(std::numeric_limits<std::uint64_t>::max()), "16 EiB");
  EXPECT_EQ(format_nbytes(1ULL << 63U), "8 EiB");
  EXPECT_EQ(format_nbytes(1024, 2, TrimZeroFraction::NO), "1.00 KiB");
  EXPECT_EQ(format_nbytes(1536, 0), "2 KiB") << "no decimals means the value is rounded";
}

TEST(StringUtilsTest, rates_scale_by_1000)
{
  // Decimal units, unlike the byte counts above, as storage throughput is quoted.
  EXPECT_EQ(format_rate(0.0), "0 B/s");
  EXPECT_EQ(format_rate(1500.0), "1.50 kB/s");
  EXPECT_EQ(format_rate(1.34e9), "1.34 GB/s");
  EXPECT_EQ(format_rate(-2.0e6), "-2 MB/s");
}

TEST(StringUtilsTest, durations_scale_through_their_units)
{
  using namespace std::chrono_literals;

  EXPECT_EQ(format_duration(0ns), "0 s");
  EXPECT_EQ(format_duration(500ns), "500 ns");
  EXPECT_EQ(format_duration(1500ns), "1.50 us");
  EXPECT_EQ(format_duration(243'803'000ns), "243.80 ms");
  EXPECT_EQ(format_duration(std::chrono::seconds{90}), "1.50 min");
  EXPECT_EQ(format_duration(std::chrono::hours{25}), "1.04 d");
  EXPECT_EQ(format_duration(-1500ns), "-1.50 us");
}
