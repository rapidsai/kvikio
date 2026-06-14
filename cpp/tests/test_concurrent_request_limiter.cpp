/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/concurrent_request_limiter.hpp>

using kvikio::detail::ConcurrentRequestLimiter;

TEST(ConcurrentRequestLimiter, EnforcesCeiling)
{
  ConcurrentRequestLimiter limiter{3};
  EXPECT_FALSE(limiter.unlimited());
  EXPECT_TRUE(limiter.try_acquire());
  EXPECT_TRUE(limiter.try_acquire());
  EXPECT_TRUE(limiter.try_acquire());
  // At capacity: further acquisitions fail without blocking.
  EXPECT_FALSE(limiter.try_acquire());
  EXPECT_FALSE(limiter.try_acquire());
}

TEST(ConcurrentRequestLimiter, ReleaseFreesSlot)
{
  ConcurrentRequestLimiter limiter{1};
  EXPECT_TRUE(limiter.try_acquire());
  EXPECT_FALSE(limiter.try_acquire());
  limiter.release();
  // The freed slot can be re-acquired, and the cap still holds at one.
  EXPECT_TRUE(limiter.try_acquire());
  EXPECT_FALSE(limiter.try_acquire());
}

TEST(ConcurrentRequestLimiter, UnlimitedAlwaysAcquires)
{
  ConcurrentRequestLimiter limiter{0};
  EXPECT_TRUE(limiter.unlimited());
  for (int i = 0; i < 10000; ++i) {
    EXPECT_TRUE(limiter.try_acquire());
  }
}

TEST(ConcurrentRequestLimiter, HardCapUnderContention)
{
  // Many threads hammer try_acquire() with no releases. The number of successes must equal the cap
  // exactly, proving the ceiling is never exceeded under concurrent acquisition.
  constexpr std::size_t cap         = 64;
  constexpr int n_threads           = 16;
  constexpr int attempts_per_thread = 10000;
  ConcurrentRequestLimiter limiter{cap};
  std::atomic<std::size_t> successes{0};

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < attempts_per_thread; ++i) {
        if (limiter.try_acquire()) { successes.fetch_add(1, std::memory_order_relaxed); }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(successes.load(), cap);
}
