/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/env.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class RemoteHandleTest : public testing::Test {
 protected:
  void SetUp() override
  {
    _sample_urls = {
      // Endpoint type: S3
      {"s3://bucket-name/object-key-name", kvikio::RemoteEndpointType::S3_PUBLIC},
      {"s3://bucket-name/object-key-name-dir/object-key-name-file",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://bucket-name.s3.region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://s3.region-code.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://bucket-name.s3.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://s3.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://bucket-name.s3-region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://s3-region-code.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},

      // Endpoint type: S3 presigned URL
      {"https://bucket-name.s3.region-code.amazonaws.com/"
       "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&X-Amz-Credential=cred&"
       "X-Amz-SignedHeaders=host",
       kvikio::RemoteEndpointType::S3_PRESIGNED_URL},

      // Endpoint type: WebHDFS
      {"https://host:1234/webhdfs/v1/data.bin", kvikio::RemoteEndpointType::WEBHDFS},
    };
  }

  void TearDown() override {}

  void test_helper(kvikio::RemoteEndpointType expected_endpoint_type,
                   std::function<bool(const std::string&)> url_validity_checker)
  {
    for (auto const& [url, endpoint_type] : _sample_urls) {
      if (endpoint_type == expected_endpoint_type) {
        // Given that the URL is the expected endpoint type

        // Test URL validity checker
        EXPECT_TRUE(url_validity_checker(url));

        // Test unified interface
        {
          // Here we pass the 1-byte argument to RemoteHandle::open. For all endpoints except
          // kvikio::RemoteEndpointType::S3, this prevents the endpoint constructor from querying
          // the file size and sending requests to the server, thus allowing us to use dummy URLs
          // for testing purpose.
          // For kvikio::RemoteEndpointType::S3, RemoteHandle::open sends HEAD request as a
          // connectivity check and will fail on the syntactically valid dummy URL. The
          // kvikio::RemoteEndpointType::S3_PUBLIC will then be used as the endpoint.
          auto remote_handle =
            kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1);
          EXPECT_EQ(remote_handle.remote_endpoint_type(), expected_endpoint_type);
        }

        // Test explicit endpoint type specification
        {
          EXPECT_NO_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, expected_endpoint_type, std::nullopt, 1);
          });
        }
      } else {
        // Given that the URL is NOT the expected endpoint type

        // Test URL validity checker
        EXPECT_FALSE(url_validity_checker(url));

        // Test explicit endpoint type specification
        {
          EXPECT_ANY_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, expected_endpoint_type, std::nullopt, 1);
          });
        }
      }
    }
  }

  std::vector<std::pair<std::string, kvikio::RemoteEndpointType>> _sample_urls;
};

TEST_F(RemoteHandleTest, s3_endpoint_constructor)
{
  kvikio::test::EnvVarContext env_var_ctx{{"AWS_DEFAULT_REGION", "my_aws_default_region"},
                                          {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
                                          {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"},
                                          {"AWS_ENDPOINT_URL", "https://my_aws_endpoint_url"}};
  std::string url        = "https://my_aws_endpoint_url/bucket_name/object_name";
  std::string aws_region = "my_aws_region";
  // Use the overload where the full url and the optional aws_region are specified.
  kvikio::S3Endpoint s1(url, aws_region);

  std::string bucket_name = "bucket_name";
  std::string object_name = "object_name";
  // Use the other overload where the bucket and object names are specified.
  kvikio::S3Endpoint s2(std::make_pair(bucket_name, object_name));

  EXPECT_EQ(s1.str(), s2.str());
}

TEST_F(RemoteHandleTest, test_http_url)
{
  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{// Incorrect scheme
                                                "s3://example.com",
                                                "hdfs://example.com",
                                                // Missing file path
                                                "http://example.com"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::HttpEndpoint::is_url_valid(invalid_url));
    }
  }
}

TEST_F(RemoteHandleTest, test_s3_url)
{
  kvikio::test::EnvVarContext env_var_ctx{{"AWS_DEFAULT_REGION", "my_aws_default_region"},
                                          {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
                                          {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"}};

  {
    test_helper(kvikio::RemoteEndpointType::S3_PUBLIC, kvikio::S3Endpoint::is_url_valid);
  }

  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{
      // Lack object-name
      "s3://bucket-name",
      "https://bucket-name.s3.region-code.amazonaws.com",
      // Presigned URL
      "https://bucket-name.s3.region-code.amazonaws.com/"
      "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&X-Amz-Credential="
      "cred&"
      "X-Amz-SignedHeaders=host"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::S3Endpoint::is_url_valid(invalid_url));
    }
  }

  // S3_PUBLIC is not in the allowlist. So when the connectivity check fails on the dummy URL,
  // KvikIO cannot fall back to S3_PUBLIC.
  {
    EXPECT_ANY_THROW({
      kvikio::RemoteHandle::open(
        "s3://bucket-name/object-key-name",
        kvikio::RemoteEndpointType::AUTO,
        std::vector<kvikio::RemoteEndpointType>{kvikio::RemoteEndpointType::S3,
                                                kvikio::RemoteEndpointType::HTTP},
        1);
    });
  }
}

TEST_F(RemoteHandleTest, test_s3_url_with_presigned_url)
{
  {
    test_helper(kvikio::RemoteEndpointType::S3_PRESIGNED_URL,
                kvikio::S3EndpointWithPresignedUrl::is_url_valid);
  }

  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{
      // Presigned URL should not use S3 scheme
      "s3://bucket-name/object-key-name",

      // Completely missing query
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name",

      // Missing key parameters ("X-Amz-..."") in query
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name?k0=v0&k1=v2"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::S3EndpointWithPresignedUrl::is_url_valid(invalid_url));
    }
  }
}

TEST_F(RemoteHandleTest, test_webhdfs_url)
{
  {
    test_helper(kvikio::RemoteEndpointType::WEBHDFS, kvikio::WebHdfsEndpoint::is_url_valid);
  }

  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{// Missing file
                                                "https://host:1234/webhdfs/v1",
                                                "https://host:1234/webhdfs/v1/",

                                                // Missing WebHDFS identifier
                                                "https://host:1234/data.bin",

                                                // Missing port number
                                                "https://host/webhdfs/v1/data.bin"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::WebHdfsEndpoint::is_url_valid(invalid_url));
    }
  }
}

TEST_F(RemoteHandleTest, test_open)
{
  // Missing scheme
  {
    std::vector<std::string> const urls{
      "example.com/path", "example.com:8080/path", "//example.com/path", "://example.com/path"};
    for (auto const& url : urls) {
      EXPECT_THROW(
        { kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1); },
        std::runtime_error);
    }
  }

  // Unsupported type
  {
    std::string const url{"unsupported://example.com/path"};
    EXPECT_THAT(
      [&] { kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1); },
      ThrowsMessage<std::runtime_error>(HasSubstr("Unsupported endpoint URL")));
  }

  // Specified URL not in the allowlist
  {
    std::string const url{"https://host:1234/webhdfs/v1/data.bin"};
    std::vector<std::vector<kvikio::RemoteEndpointType>> const wrong_allowlists{
      {},
      {kvikio::RemoteEndpointType::S3},
    };
    for (auto const& wrong_allowlist : wrong_allowlists) {
      EXPECT_THAT(
        [&] {
          kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::WEBHDFS, wrong_allowlist, 1);
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("is not in the allowlist")));
    }
  }

  // Invalid URLs
  {
    std::vector<std::pair<std::string, kvikio::RemoteEndpointType>> const invalid_urls{
      {"s3://bucket-name", kvikio::RemoteEndpointType::S3},
      {"https://bucket-name.s3.region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PRESIGNED_URL},
      {"https://host:1234/webhdfs/v1", kvikio::RemoteEndpointType::WEBHDFS},
      {"http://example.com", kvikio::RemoteEndpointType::HTTP},
    };
    for (auto const& [invalid_url, endpoint_type] : invalid_urls) {
      EXPECT_THAT([&] { kvikio::RemoteHandle::open(invalid_url, endpoint_type, std::nullopt, 1); },
                  ThrowsMessage<std::runtime_error>(HasSubstr("Invalid URL")));
    }
  }
}
