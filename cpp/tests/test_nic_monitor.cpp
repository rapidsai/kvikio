/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include <gtest/gtest.h>

#include "nic_monitor.hpp"

using kvikio::nsys_plugin::compute_rates;
using kvikio::nsys_plugin::default_interfaces;
using kvikio::nsys_plugin::iface_is_up;
using kvikio::nsys_plugin::NicCounterReader;
using kvikio::nsys_plugin::NicCounters;
using kvikio::nsys_plugin::constants::bytes_per_mib;

TEST(NicMonitor, ReaderReadsLoopback)
{
  NicCounterReader const reader{{"lo"}};
  auto const counters = reader.read();
  ASSERT_EQ(counters.size(), 1U);
  // The loopback interface is present on essentially every Linux host.
  EXPECT_TRUE(counters[0].has_value());
}

TEST(NicMonitor, ReaderReportsMissingInterfaceWithoutValue)
{
  NicCounterReader const reader{{"kvikio_no_such_iface"}};
  auto const counters = reader.read();
  ASSERT_EQ(counters.size(), 1U);
  EXPECT_FALSE(counters[0].has_value());
}

TEST(NicMonitor, IfaceIsUpAcceptsLoopbackAndRejectsMissing)
{
  // Loopback reports operstate "unknown", which the up/unknown policy accepts.
  EXPECT_TRUE(iface_is_up("lo"));
  // A name that cannot be read must be treated as not up rather than accepted.
  EXPECT_FALSE(iface_is_up("kvikio_no_such_iface"));
}

TEST(NicMonitor, DefaultInterfacesExcludeLoopback)
{
  for (auto const& name : default_interfaces()) {
    EXPECT_NE(name, "lo");
  }
}

TEST(NicMonitor, ComputeRatesConvertsDeltasToMiBps)
{
  NicCounters const prev{0, 0};
  NicCounters const cur{static_cast<std::uint64_t>(bytes_per_mib) * 2,
                        static_cast<std::uint64_t>(bytes_per_mib) * 6};
  auto const rates = compute_rates(prev, cur, 2.0);
  // 2 MiB received over 2 s -> 1 MiB/s; 6 MiB transmitted over 2 s -> 3 MiB/s.
  EXPECT_DOUBLE_EQ(rates.rx, 1.0);
  EXPECT_DOUBLE_EQ(rates.tx, 3.0);
}

TEST(NicMonitor, ComputeRatesGuardsWrapAndNonPositiveInterval)
{
  NicCounters const high{1000, 1000};
  NicCounters const low{10, 10};
  // A counter that appears to go backwards (wrap or reset) yields zero for that direction.
  auto const wrapped = compute_rates(high, low, 1.0);
  EXPECT_DOUBLE_EQ(wrapped.rx, 0.0);
  EXPECT_DOUBLE_EQ(wrapped.tx, 0.0);
  // A non-positive elapsed time yields zero rather than dividing by zero.
  auto const no_time = compute_rates(low, high, 0.0);
  EXPECT_DOUBLE_EQ(no_time.rx, 0.0);
  EXPECT_DOUBLE_EQ(no_time.tx, 0.0);
}
