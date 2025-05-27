/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/defaults.hpp>

#include "utils/env.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(DefaultsTest, parse_compat_mode_str)
{
  {
    std::vector<std::string> inputs{
      "ON", "on", "On", "TRUE", "true", "True", "YES", "yes", "Yes", "1"};
    for (auto const& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::ON);
    }
  }

  {
    std::vector<std::string> inputs{
      "OFF", "off", "oFf", "FALSE", "false", "False", "NO", "no", "No", "0"};
    for (auto const& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::OFF);
    }
  }

  {
    std::vector<std::string> inputs{"AUTO", "auto", "aUtO"};
    for (auto const& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::AUTO);
    }
  }

  {
    std::vector<std::string> inputs{"", "invalidOption", "11", "*&^Yes"};
    for (auto const& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_compat_mode_str(input), std::invalid_argument);
    }
  }
}

TEST(DefaultsTest, parse_http_status_codes)
{
  {
    std::vector<std::string> inputs{
      "429,500", "429, 500", " 429,500", "429,  500", "429 ,500", "429,500 "};
    std::vector<int> expected = {429, 500};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_http_status_codes("KVIKIO_HTTP_STATUS_CODES", input),
                expected);
    }
  }

  {
    std::vector<std::string> inputs{"429,", ",429", "a,b", "429,,500", "429,1000"};
    for (const auto& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_http_status_codes("KVIKIO_HTTP_STATUS_CODES", input),
                   std::invalid_argument);
    }
  }
}

TEST(DefaultsTest, alias_for_getenv_or)
{
  // Passed initializer list is empty
  {
    EXPECT_THAT([=] { kvikio::getenv_or({}, 123); },
                ThrowsMessage<std::invalid_argument>(HasSubstr(
                  "`env_var_names` must contain at least one environment variable name")));
  }

  // Env var has an empty value
  {
    kvikio::test::EnvVarContext env_var_ctx{{{"KVIKIO_TEST_ALIAS", ""}}};
    EXPECT_THAT([=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, 123); },
                ThrowsMessage<std::invalid_argument>(
                  HasSubstr("KVIKIO_TEST_ALIAS must not have an empty value")));
  }

  // Env var has already been set by its alias
  {
    kvikio::test::EnvVarContext env_var_ctx{
      {{"KVIKIO_TEST_ALIAS_1", "10"}, {"KVIKIO_TEST_ALIAS_2", "20"}}};
    EXPECT_THAT([=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123); },
                ThrowsMessage<std::invalid_argument>(HasSubstr(
                  "Environment variable KVIKIO_TEST_ALIAS_2 has already been set by its alias")));
  }

  // Env var has invalid value
  {
    kvikio::test::EnvVarContext env_var_ctx{{{"KVIKIO_TEST_ALIAS", "abc"}}};
    EXPECT_THAT([=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, 123); },
                ThrowsMessage<std::invalid_argument>(
                  HasSubstr("Unknown config value KVIKIO_TEST_ALIAS=abc")));
  }

  // 1st alias has a set value
  {
    kvikio::test::EnvVarContext env_var_ctx{{{"KVIKIO_TEST_ALIAS_1", "654.321"}}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123.456);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS_1"});
    EXPECT_EQ(result, 654.321);
    EXPECT_TRUE(has_found);
  }

  // 2nd alias has a set value
  {
    kvikio::test::EnvVarContext env_var_ctx{{{"KVIKIO_TEST_ALIAS_2", "654.321"}}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123.456);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS_2"});
    EXPECT_EQ(result, 654.321);
    EXPECT_TRUE(has_found);
  }

  // Neither alias has a set value
  {
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123.456);
    EXPECT_TRUE(env_var_name.empty());
    EXPECT_EQ(result, 123.456);
    EXPECT_FALSE(has_found);
  }
}
