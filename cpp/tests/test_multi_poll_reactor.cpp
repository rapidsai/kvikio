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
constexpr std::string_view kDispatchEnv = "KVIKIO_REMOTE_IO_REACTOR_DISPATCH";

}  // namespace

TEST(RemoteIOBackendParse, RecognizedNamesEasyThreadpool)
{
  // Only the canonical name is accepted. Case-insensitive plus leading/trailing whitespace is fine.
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
  unsetenv(std::string{kBackendEnv}.c_str());
  EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
            kvikio::RemoteIOBackend::EASY_THREADPOOL);
  EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::MULTI_POLL),
            kvikio::RemoteIOBackend::MULTI_POLL);
}

TEST(RemoteIOBackendParse, BadValueThrows)
{
  // The short aliases "easy" and "multi" are deliberately rejected.
  for (auto const& v : {"bogus", "easy", "multi", "easythreadpool", "multipoll", ""}) {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, v}};
    EXPECT_THROW(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
                 std::invalid_argument)
      << "value: " << v;
  }
}

TEST(RemoteReactorDispatchParse, RecognizedNames)
{
  for (auto const& v : {"per_chunk", "  Per_Chunk  "}) {
    kvikio::test::EnvVarContext ctx{{kDispatchEnv, v}};
    EXPECT_EQ(kvikio::getenv_or(kDispatchEnv, kvikio::RemoteReactorDispatch::PER_PREAD),
              kvikio::RemoteReactorDispatch::PER_CHUNK)
      << "value: " << v;
  }

  for (auto const& v : {"per_pread", "PER_PREAD"}) {
    kvikio::test::EnvVarContext ctx{{kDispatchEnv, v}};
    EXPECT_EQ(kvikio::getenv_or(kDispatchEnv, kvikio::RemoteReactorDispatch::PER_CHUNK),
              kvikio::RemoteReactorDispatch::PER_PREAD)
      << "value: " << v;
  }
}

TEST(RemoteReactorDispatchParse, UnsetReturnsDefault)
{
  unsetenv(std::string{kDispatchEnv}.c_str());
  EXPECT_EQ(kvikio::getenv_or(kDispatchEnv, kvikio::RemoteReactorDispatch::PER_CHUNK),
            kvikio::RemoteReactorDispatch::PER_CHUNK);
  EXPECT_EQ(kvikio::getenv_or(kDispatchEnv, kvikio::RemoteReactorDispatch::PER_PREAD),
            kvikio::RemoteReactorDispatch::PER_PREAD);
}

TEST(RemoteReactorDispatchParse, BadValueThrows)
{
  for (auto const& v : {"bogus", "per_byte", "round_robin", ""}) {
    kvikio::test::EnvVarContext ctx{{kDispatchEnv, v}};
    EXPECT_THROW(kvikio::getenv_or(kDispatchEnv, kvikio::RemoteReactorDispatch::PER_CHUNK),
                 std::invalid_argument)
      << "value: " << v;
  }
}
