/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <kvikio/statistics/counters.hpp>

using kvikio::statistics::Counters;
using kvikio::statistics::counters;
using kvikio::statistics::ReportRows;

TEST(CountersTest, every_counter_is_differenced_and_none_of_them_wrap)
{
  // A reading of the live counters differences against itself, whatever the process has done.
  auto const now = counters();
  EXPECT_TRUE(now.since(now).empty());

  // `since()` names each counter, so this checks that none was missed. Every field is eight bytes
  // wide, which is what lets the whole struct be filled a word at a time.
  constexpr auto words = sizeof(Counters) / sizeof(std::uint64_t);
  std::array<std::uint64_t, words> earlier{};
  std::array<std::uint64_t, words> later{};
  for (std::size_t i = 0; i < words; ++i) {
    earlier[i] = i + 1;
    later[i]   = (i + 1) * 10;
  }

  auto const spent = std::bit_cast<Counters>(later).since(std::bit_cast<Counters>(earlier));
  auto const got   = std::bit_cast<std::array<std::uint64_t, words>>(spent);
  for (std::size_t i = 0; i < words; ++i) {
    EXPECT_EQ(got[i], later[i] - earlier[i]) << "counter " << i << " was not differenced";
  }

  // Backwards, every counter saturates rather than wrapping.
  EXPECT_TRUE(std::bit_cast<Counters>(earlier).since(std::bit_cast<Counters>(later)).empty());

  // And the words really are the named counters, not some other layout.
  Counters one;
  one.http_retries = 7;
  EXPECT_EQ(one.since(Counters{}).http_retries, 7);
  EXPECT_EQ(one.since(Counters{}).http_connections, 0);
}

TEST(CountersTest, a_reading_formats_only_what_was_paid_for)
{
  Counters nothing;
  EXPECT_TRUE(nothing.empty());
  EXPECT_EQ(nothing.report(), "") << "a reading of nothing should say nothing";
  // Unless everything is asked for, which is how to see what can be recorded at all.
  EXPECT_NE(nothing.report(ReportRows::ALL).find("http"), std::string::npos);

  Counters reading;
  reading.remote_size_probes  = 12;
  reading.remote_size_probing = std::chrono::milliseconds{600};
  reading.http_connections    = 128;
  reading.http_dns            = std::chrono::milliseconds{40};
  reading.http_tcp            = std::chrono::milliseconds{900};
  reading.http_tls            = std::chrono::milliseconds{1900};
  reading.http_retries        = 7;
  reading.http_retry_backoff  = std::chrono::milliseconds{1200};

  auto const text = reading.report();
  EXPECT_NE(text.find("http size probes     12 probes, 600 ms"), std::string::npos);
  EXPECT_NE(text.find("128 connections, 40 ms dns, 900 ms tcp, 1.90 s tls"), std::string::npos);
  EXPECT_NE(text.find("http retries         7 retries, 1.20 s backoff"), std::string::npos);

  auto const json = reading.to_json();
  EXPECT_NE(json.find("\"http_retries\": 7"), std::string::npos);
  EXPECT_NE(json.find("\"http_retry_backoff_ns\": 1200000000"), std::string::npos);
  // The JSON schema is the same whatever was used, unlike the report.
  EXPECT_NE(nothing.to_json().find("\"http_connections\": 0"), std::string::npos);
}
