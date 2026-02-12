/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <gmock/gmock.h>

#include <kvikio/detail/utils.hpp>

using namespace kvikio::detail;

TEST(ScopeExitTest, invokes_on_normal_exit)
{
  int x = 0;
  {
    ScopeExit guard([&]() { x = 2077; });
  }
  EXPECT_EQ(x, 2077);
}

TEST(ScopeExitTest, invokes_on_exception)
{
  int x = 0;
  try {
    ScopeExit guard([&]() { x = 2077; });
    throw std::runtime_error("Error");
  } catch (...) {
  }
  EXPECT_EQ(x, 2077);
}

TEST(ScopeExitTest, captures_by_reference)
{
  int a = 1, b = 2;
  {
    ScopeExit guard([&]() { std::swap(a, b); });
  }
  EXPECT_EQ(a, 2);
  EXPECT_EQ(b, 1);
}

TEST(ScopeExitTest, propagates_cleanup_exception_on_normal_exit)
{
  EXPECT_THROW(
    { ScopeExit guard([]() { throw std::runtime_error("Cleanup failed"); }); }, std::runtime_error);
}

TEST(ScopeExitTest, suppresses_cleanup_exception_during_unwinding)
{
  EXPECT_THROW(
    {
      ScopeExit guard([]() { throw std::runtime_error("Cleanup failed"); });
      throw std::logic_error("Original exception");
    },
    std::logic_error);  // Original exception survives, cleanup exception suppressed
}
