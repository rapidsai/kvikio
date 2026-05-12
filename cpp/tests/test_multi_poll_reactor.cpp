/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/env.hpp"

namespace {

constexpr std::string_view kBackendEnv  = "KVIKIO_REMOTE_IO_BACKEND";
constexpr std::string_view kShardingEnv = "KVIKIO_REMOTE_IO_REACTOR_SHARDING";

}  // namespace

// The integration scenarios for the MULTI_POLL backend (range reads, concurrency, error
// handling) are covered by the Python pytest suite re-run under
// `KVIKIO_REMOTE_IO_BACKEND=multi_poll`. The test cases here cover plumbing the pytest
// run cannot reach cleanly: env-var parsing of the two new enums.

TEST(RemoteIOBackendParse, RecognizedNamesEasyThreadpool)
{
  // Only the canonical name is accepted; case-insensitive plus leading/trailing
  // whitespace is fine.
  for (auto const& v : {"easy_threadpool", "EASY_THREADPOOL", "  easy_threadpool  "}) {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, v}};
    EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::MULTI_POLL),
              kvikio::RemoteIOBackend::EASY_THREADPOOL)
      << "value: " << v;
  }
}

TEST(RemoteIOBackendParse, RecognizedNamesMultiPoll)
{
  for (auto const& v : {"multi_poll", "MULTI_POLL", "Multi_Poll", "  multi_poll  "}) {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, v}};
    EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
              kvikio::RemoteIOBackend::MULTI_POLL)
      << "value: " << v;
  }
}

TEST(RemoteIOBackendParse, UnsetReturnsDefault)
{
  // No EnvVarContext: assume the variable is unset (or, if set in the test environment,
  // the parsing path is exercised; either way the default fallback path is implicit in
  // the unset case).
  unsetenv(std::string{kBackendEnv}.c_str());
  EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
            kvikio::RemoteIOBackend::EASY_THREADPOOL);
  EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::MULTI_POLL),
            kvikio::RemoteIOBackend::MULTI_POLL);
}

TEST(RemoteIOBackendParse, BadValueThrows)
{
  // The short aliases "easy" and "multi" are deliberately rejected; they would be
  // ambiguous once MULTI_SOCKET lands.
  for (auto const& v :
       {"bogus", "easy", "multi", "multi_socket", "easythreadpool", "multipoll", ""}) {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, v}};
    EXPECT_THROW(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
                 std::invalid_argument)
      << "value: " << v;
  }
}

TEST(RemoteReactorShardingParse, RecognizedNames)
{
  {
    kvikio::test::EnvVarContext ctx{{kShardingEnv, "per_chunk"}};
    EXPECT_EQ(kvikio::getenv_or(kShardingEnv, kvikio::RemoteReactorSharding::PER_PREAD),
              kvikio::RemoteReactorSharding::PER_CHUNK);
  }
  {
    kvikio::test::EnvVarContext ctx{{kShardingEnv, "PER_PREAD"}};
    EXPECT_EQ(kvikio::getenv_or(kShardingEnv, kvikio::RemoteReactorSharding::PER_CHUNK),
              kvikio::RemoteReactorSharding::PER_PREAD);
  }
  {
    kvikio::test::EnvVarContext ctx{{kShardingEnv, "  Per_Chunk  "}};
    EXPECT_EQ(kvikio::getenv_or(kShardingEnv, kvikio::RemoteReactorSharding::PER_PREAD),
              kvikio::RemoteReactorSharding::PER_CHUNK);
  }
}

TEST(RemoteReactorShardingParse, UnsetReturnsDefault)
{
  unsetenv(std::string{kShardingEnv}.c_str());
  EXPECT_EQ(kvikio::getenv_or(kShardingEnv, kvikio::RemoteReactorSharding::PER_CHUNK),
            kvikio::RemoteReactorSharding::PER_CHUNK);
  EXPECT_EQ(kvikio::getenv_or(kShardingEnv, kvikio::RemoteReactorSharding::PER_PREAD),
            kvikio::RemoteReactorSharding::PER_PREAD);
}

TEST(RemoteReactorShardingParse, BadValueThrows)
{
  for (auto const& v : {"bogus", "per_byte", "round_robin", ""}) {
    kvikio::test::EnvVarContext ctx{{kShardingEnv, v}};
    EXPECT_THROW(kvikio::getenv_or(kShardingEnv, kvikio::RemoteReactorSharding::PER_CHUNK),
                 std::invalid_argument)
      << "value: " << v;
  }
}
