/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/experimental/nic_monitor.hpp>

using kvikio::experimental::NicBandwidthMonitor;
using kvikio::experimental::read_nic_counters;

TEST(NicMonitor, ReadCountersIncludesLoopback)
{
  auto counters = read_nic_counters();
  // The loopback interface is present on essentially every Linux host.
  ASSERT_TRUE(counters.find("lo") != counters.end());
}

TEST(NicMonitor, RejectsNonPositiveFrequency)
{
  EXPECT_THROW(NicBandwidthMonitor(0.0), std::invalid_argument);
  EXPECT_THROW(NicBandwidthMonitor(-1.0), std::invalid_argument);
}

TEST(NicMonitor, StartStopExplicitInterface)
{
  NicBandwidthMonitor monitor{200.0, std::vector<std::string>{"lo"}};
  EXPECT_FALSE(monitor.running());
  monitor.start();
  EXPECT_TRUE(monitor.running());
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  monitor.stop();
  EXPECT_FALSE(monitor.running());
  EXPECT_EQ(monitor.interfaces().size(), 1U);
}

TEST(NicMonitor, DefaultSelectsInterfacesAndStopIsIdempotent)
{
  NicBandwidthMonitor monitor{100.0};
  monitor.start();
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  // `lo` is excluded from the default set; the monitor still runs cleanly.
  for (auto const& name : monitor.interfaces()) {
    EXPECT_NE(name, "lo");
  }
  monitor.stop();
  monitor.stop();  // Safe to call more than once.
  EXPECT_FALSE(monitor.running());
}
