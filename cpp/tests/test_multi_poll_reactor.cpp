/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/env.hpp"

namespace {

constexpr std::string_view backend_env  = "KVIKIO_REMOTE_IO_BACKEND";
constexpr std::string_view dispatch_env = "KVIKIO_REMOTE_IO_REACTOR_DISPATCH";

}  // namespace

TEST(RemoteIOBackendParse, recognized_names_easy_threadpool)
{
  // Only the canonical name is accepted. Case-insensitive plus leading/trailing whitespace is fine.
  for (auto const& v : {"easy_threadpool", "EASY_THREADPOOL", "  easy_threadpool  "}) {
    kvikio::test::EnvVarContext ctx{{backend_env, v}};
    EXPECT_EQ(kvikio::getenv_or(backend_env, kvikio::RemoteIOBackend::MULTI_POLL),
              kvikio::RemoteIOBackend::EASY_THREADPOOL)
      << "value: " << v;
  }
}

TEST(RemoteIOBackendParse, recognized_names_multi_poll)
{
  for (auto const& v : {"multi_poll", "MULTI_POLL", "Multi_Poll", "  multi_poll  "}) {
    kvikio::test::EnvVarContext ctx{{backend_env, v}};
    EXPECT_EQ(kvikio::getenv_or(backend_env, kvikio::RemoteIOBackend::EASY_THREADPOOL),
              kvikio::RemoteIOBackend::MULTI_POLL)
      << "value: " << v;
  }
}

TEST(RemoteIOBackendParse, unset_returns_default)
{
  unsetenv(std::string{backend_env}.c_str());
  EXPECT_EQ(kvikio::getenv_or(backend_env, kvikio::RemoteIOBackend::EASY_THREADPOOL),
            kvikio::RemoteIOBackend::EASY_THREADPOOL);
  EXPECT_EQ(kvikio::getenv_or(backend_env, kvikio::RemoteIOBackend::MULTI_POLL),
            kvikio::RemoteIOBackend::MULTI_POLL);
}

TEST(RemoteIOBackendParse, bad_value_throws)
{
  // The short aliases "easy" and "multi" are deliberately rejected.
  for (auto const& v : {"bogus", "easy", "multi", "easythreadpool", "multipoll", ""}) {
    kvikio::test::EnvVarContext ctx{{backend_env, v}};
    EXPECT_THROW(kvikio::getenv_or(backend_env, kvikio::RemoteIOBackend::EASY_THREADPOOL),
                 std::invalid_argument)
      << "value: " << v;
  }
}

TEST(RemoteReactorDispatchParse, recognized_names)
{
  for (auto const& v : {"per_chunk", "  Per_Chunk  "}) {
    kvikio::test::EnvVarContext ctx{{dispatch_env, v}};
    EXPECT_EQ(kvikio::getenv_or(dispatch_env, kvikio::RemoteReactorDispatch::PER_PREAD),
              kvikio::RemoteReactorDispatch::PER_CHUNK)
      << "value: " << v;
  }

  for (auto const& v : {"per_pread", "PER_PREAD"}) {
    kvikio::test::EnvVarContext ctx{{dispatch_env, v}};
    EXPECT_EQ(kvikio::getenv_or(dispatch_env, kvikio::RemoteReactorDispatch::PER_CHUNK),
              kvikio::RemoteReactorDispatch::PER_PREAD)
      << "value: " << v;
  }
}

TEST(RemoteReactorDispatchParse, unset_returns_default)
{
  unsetenv(std::string{dispatch_env}.c_str());
  EXPECT_EQ(kvikio::getenv_or(dispatch_env, kvikio::RemoteReactorDispatch::PER_CHUNK),
            kvikio::RemoteReactorDispatch::PER_CHUNK);
  EXPECT_EQ(kvikio::getenv_or(dispatch_env, kvikio::RemoteReactorDispatch::PER_PREAD),
            kvikio::RemoteReactorDispatch::PER_PREAD);
}

TEST(RemoteReactorDispatchParse, bad_value_throws)
{
  for (auto const& v : {"bogus", "per_byte", "round_robin", ""}) {
    kvikio::test::EnvVarContext ctx{{dispatch_env, v}};
    EXPECT_THROW(kvikio::getenv_or(dispatch_env, kvikio::RemoteReactorDispatch::PER_CHUNK),
                 std::invalid_argument)
      << "value: " << v;
  }
}

TEST(ConnectionCacheSize, leaves_headroom_over_the_ceiling)
{
  EXPECT_EQ(kvikio::detail::connection_cache_size(std::optional<std::size_t>{8}).value(), 32L);
  EXPECT_EQ(kvikio::detail::connection_cache_size(std::optional<std::size_t>{128}).value(), 512L);
}

TEST(ConnectionCacheSize, tiny_ceilings_stay_usable)
{
  EXPECT_EQ(kvikio::detail::connection_cache_size(std::optional<std::size_t>{1}).value(), 4L);
  EXPECT_EQ(kvikio::detail::connection_cache_size(std::optional<std::size_t>{0}).value(), 4L);
}

TEST(ConnectionCacheSize, unlimited_concurrency_leaves_the_default_alone)
{
  EXPECT_FALSE(kvikio::detail::connection_cache_size(std::nullopt).has_value());
}

TEST(ConnectionCacheSize, never_exceeds_what_libcurl_accepts)
{
  constexpr auto limit = std::min(static_cast<long>(std::numeric_limits<unsigned>::max()),
                                  std::numeric_limits<long>::max());
  std::array ceilings  = {std::numeric_limits<std::size_t>::max(),
                          std::numeric_limits<std::size_t>::max() / 2,
                          static_cast<std::size_t>(limit),
                          static_cast<std::size_t>(limit) / 2 + 1};
  for (auto const ceiling : ceilings) {
    auto const size = kvikio::detail::connection_cache_size(std::optional<std::size_t>{ceiling});
    ASSERT_TRUE(size.has_value()) << "ceiling: " << ceiling;
    EXPECT_LE(size.value(), limit) << "ceiling: " << ceiling;
    EXPECT_GT(size.value(), 0L) << "ceiling: " << ceiling;
  }
}
