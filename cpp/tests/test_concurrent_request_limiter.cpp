/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/concurrent_request_limiter.hpp>

using kvikio::detail::ConcurrentRequestLimiter;

TEST(ConcurrentRequestLimiter, EnforcesCeiling)
{
  ConcurrentRequestLimiter limiter{3};
  EXPECT_FALSE(limiter.unlimited());
  auto s1 = limiter.try_acquire();
  auto s2 = limiter.try_acquire();
  auto s3 = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(s1));
  EXPECT_TRUE(static_cast<bool>(s2));
  EXPECT_TRUE(static_cast<bool>(s3));
  // At capacity: further acquisitions return empty slots without blocking.
  EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));
  EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));
}

TEST(ConcurrentRequestLimiter, SlotScopeReleases)
{
  ConcurrentRequestLimiter limiter{1};
  {
    auto s = limiter.try_acquire();
    EXPECT_TRUE(static_cast<bool>(s));
    EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));
  }
  auto s = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(s));
  EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));
}

TEST(ConcurrentRequestLimiter, SlotMoveTransfersOwnership)
{
  ConcurrentRequestLimiter limiter{1};
  auto src = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(src));

  // Move construction: dst holds the reservation, src is empty.
  auto dst = std::move(src);
  EXPECT_TRUE(static_cast<bool>(dst));
  EXPECT_FALSE(static_cast<bool>(src));

  // Still at capacity: exactly one live reservation exists after the move.
  EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));

  // Dropping dst frees the single slot.
  dst.reset();
  EXPECT_TRUE(static_cast<bool>(limiter.try_acquire()));
}

TEST(ConcurrentRequestLimiter, SlotMoveAssignReleasesTarget)
{
  ConcurrentRequestLimiter limiter{2};
  auto a = limiter.try_acquire();
  auto b = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(a));
  EXPECT_TRUE(static_cast<bool>(b));
  EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));

  // Assigning b over a releases a's own reservation first, freeing one slot.
  a = std::move(b);
  EXPECT_TRUE(static_cast<bool>(a));
  EXPECT_FALSE(static_cast<bool>(b));
  auto c = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(c));
}

TEST(ConcurrentRequestLimiter, SlotResetIsIdempotent)
{
  ConcurrentRequestLimiter limiter{1};
  auto s = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(s));
  s.reset();
  EXPECT_FALSE(static_cast<bool>(s));
  // A second reset is a no-op.
  s.reset();
  // Exactly one slot exists: acquire succeeds once, then the ceiling holds.
  auto s2 = limiter.try_acquire();
  EXPECT_TRUE(static_cast<bool>(s2));
  EXPECT_FALSE(static_cast<bool>(limiter.try_acquire()));
}

TEST(ConcurrentRequestLimiter, UnlimitedAlwaysAcquires)
{
  constexpr int n_slots = 10000;
  ConcurrentRequestLimiter limiter{std::nullopt};
  EXPECT_TRUE(limiter.unlimited());
  std::vector<ConcurrentRequestLimiter::Slot> held;
  held.reserve(n_slots);
  for (int i = 0; i < n_slots; ++i) {
    auto s = limiter.try_acquire();
    EXPECT_TRUE(static_cast<bool>(s));
    held.push_back(std::move(s));
  }
  // All held slots are returned to the limiter when `held` destructs.
}

TEST(ConcurrentRequestLimiter, HardCapUnderContention)
{
  // Many threads hammer try_acquire() and keep every slot they win until the end of the test. The
  // number of engaged slots handed out must equal the cap exactly, proving the ceiling is never
  // exceeded under concurrent acquisition.
  constexpr std::size_t cap         = 64;
  constexpr int n_threads           = 16;
  constexpr int attempts_per_thread = 10000;
  ConcurrentRequestLimiter limiter{cap};
  std::atomic<std::size_t> successes{0};

  std::vector<std::vector<ConcurrentRequestLimiter::Slot>> held(n_threads);

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      auto& mine = held[t];
      mine.reserve(cap);
      for (int i = 0; i < attempts_per_thread; ++i) {
        auto s = limiter.try_acquire();
        if (s) {
          successes.fetch_add(1, std::memory_order_relaxed);
          mine.push_back(std::move(s));
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(successes.load(), cap);
}
