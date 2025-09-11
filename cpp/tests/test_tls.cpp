/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/detail/tls.hpp>

#include "utils/env.hpp"

TEST(TlsTest, get_ca_paths)
{
  std::string const expected_ca_bundle_path{"ca_bundle_path"};
  std::string const expected_ca_directory{"ca_directory"};
  {
    // Env var CURL_CA_BUNDLE has the highest priority. Both SSL_CERT_FILE and SSL_CERT_DIR shall be
    // skipped
    kvikio::test::EnvVarContext env_var_ctx{{"CURL_CA_BUNDLE", expected_ca_bundle_path},
                                            {"SSL_CERT_FILE", "another_ca_bundle_path"},
                                            {"SSL_CERT_DIR", expected_ca_directory}};
    auto const& [ca_bundle_file, ca_directory] = kvikio::detail::get_ca_paths();

    EXPECT_EQ(ca_bundle_file, expected_ca_bundle_path);
    EXPECT_EQ(ca_directory, std::nullopt);
  }

  {
    // Env var CURL_CA_BUNDLE and SSL_CERT_FILE are not specified, SSL_CERT_DIR shall be used
    kvikio::test::EnvVarContext env_var_ctx{{"SSL_CERT_DIR", expected_ca_directory}};
    auto const& [ca_bundle_file, ca_directory] = kvikio::detail::get_ca_paths();

    EXPECT_EQ(ca_bundle_file, std::nullopt);
    EXPECT_EQ(ca_directory, expected_ca_directory);
  }
}
