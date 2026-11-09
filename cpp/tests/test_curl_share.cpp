/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/curl_share.hpp>

using kvikio::detail::CurlShareRegistry;

TEST(CurlShareRegistryTest, a_cache_is_filled_before_another_is_created)
{
  CurlShareRegistry registry{2};
  EXPECT_EQ(registry.num_caches(), std::size_t{0});

  auto* const first  = registry.acquire();
  auto* const second = registry.acquire();
  EXPECT_EQ(registry.num_caches(), std::size_t{1});
  EXPECT_EQ(first, second) << "threads below the limit share one cache";

  auto* const third = registry.acquire();
  EXPECT_EQ(registry.num_caches(), std::size_t{2}) << "a full cache forces a new one";
  EXPECT_NE(third, first);
}

TEST(CurlShareRegistryTest, a_released_slot_is_reused)
{
  CurlShareRegistry registry{1};
  auto* const first = registry.acquire();
  registry.release(first);

  auto* const second = registry.acquire();
  EXPECT_EQ(registry.num_caches(), std::size_t{1}) << "the free slot is taken instead of adding";
  EXPECT_EQ(second, first);
}

TEST(CurlShareRegistryTest, concurrent_threads_do_not_oversubscribe_a_cache)
{
  constexpr std::size_t num_threads           = 8;
  constexpr std::size_t max_threads_per_cache = 3;
  CurlShareRegistry registry{max_threads_per_cache};

  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&registry] { std::ignore = registry.acquire(); });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  // A cache is added only once every existing one is full.
  EXPECT_EQ(registry.num_caches(), std::size_t{3});
}

TEST(CurlShareRegistryTest, a_cache_must_hold_at_least_one_thread)
{
  EXPECT_THROW(CurlShareRegistry{0}, std::invalid_argument);
}
