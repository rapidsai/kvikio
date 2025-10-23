/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
