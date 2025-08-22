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

#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/remote_handle.hpp>

#include "kvikio/hdfs.hpp"
#include "utils/env.hpp"

class RemoteHandleTest : public testing::Test {
 protected:
  void SetUp() override
  {
    _urls_and_endpoint_type = {
      // Endpoint type: S3
      {"s3://bucket-name/object-key-name", kvikio::RemoteEndpointType::S3},
      {"https://bucket-name.s3.region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3},
      {"https://s3.region-code.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3},
      {"https://bucket-name.s3.amazonaws.com/object-key-name", kvikio::RemoteEndpointType::S3},
      {"https://s3.amazonaws.com/bucket-name/object-key-name", kvikio::RemoteEndpointType::S3},
      {"https://bucket-name.s3-region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3},
      {"https://s3-region-code.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3},

      // Endpoint type: S3 presigned URL
      {"https://bucket-name.s3.region-code.amazonaws.com/"
       "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&X-Amz-Credential=cred&"
       "X-Amz-SignedHeaders=host",
       kvikio::RemoteEndpointType::S3_PRESIGNED_URL},

      // Endpoint type: WebHDFS
      {"https://host:1234/webhdfs/v1/data.bin", kvikio::RemoteEndpointType::WEBHDFS},

      // Endpoint type: HTTP
      {"https://host.com/data.bin", kvikio::RemoteEndpointType::HTTP},
    };
  }

  void TearDown() override {}

  std::vector<std::pair<std::string, kvikio::RemoteEndpointType>> _urls_and_endpoint_type;
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

TEST_F(RemoteHandleTest, check_http_url)
{
  // Invalid URLs
  {
    std::vector<std::string> urls{"s3://example.com", "hdfs://example.com"};
    for (auto const& url : urls) {
      EXPECT_FALSE(kvikio::HttpEndpoint::is_url_valid(url));
    }
  }
}

TEST_F(RemoteHandleTest, check_s3_url)
{
  kvikio::test::EnvVarContext env_var_ctx{{"AWS_DEFAULT_REGION", "my_aws_default_region"},
                                          {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
                                          {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"}};

  {
    for (auto const& [url, expected_endpoint_type] : _urls_and_endpoint_type) {
      if (expected_endpoint_type == kvikio::RemoteEndpointType::S3) {
        // Given that the URL is a valid S3 URL

        // Test S3 URL validity checker
        EXPECT_TRUE(kvikio::S3Endpoint::is_url_valid(url));

        // Test unified interface
        {
          auto remote_handle =
            kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1);
          EXPECT_EQ(remote_handle.type(), kvikio::RemoteEndpointType::S3);
        }

        // Test explicit endpoint type specification
        {
          EXPECT_NO_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::S3, std::nullopt, 1);
          });
        }
      } else {
        // Given that the URL is not an S3 URL

        // Test S3 URL validity checker
        EXPECT_FALSE(kvikio::S3Endpoint::is_url_valid(url));

        // Test explicit endpoint type specification
        {
          EXPECT_ANY_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::S3, std::nullopt, 1);
          });
        }
      }
    }
  }
}

TEST_F(RemoteHandleTest, check_s3_url_with_presigned_url)
{
  {
    for (auto const& [url, expected_endpoint_type] : _urls_and_endpoint_type) {
      if (expected_endpoint_type == kvikio::RemoteEndpointType::S3_PRESIGNED_URL) {
        // Given that the URL is a valid S3 pre-signed URL

        // Test S3 URL validity checker
        EXPECT_TRUE(kvikio::S3EndpointWithPresignedUrl::is_url_valid(url));

        // Test unified interface
        {
          auto remote_handle =
            kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1);
          EXPECT_EQ(remote_handle.type(), kvikio::RemoteEndpointType::S3_PRESIGNED_URL);
        }

        // Test explicit endpoint type specification
        {
          EXPECT_NO_THROW({
            auto remote_handle = kvikio::RemoteHandle::open(
              url, kvikio::RemoteEndpointType::S3_PRESIGNED_URL, std::nullopt, 1);
          });
        }
      } else {
        // Given that the URL is not an S3 pre-signed URL

        // Test S3 pre-signed URL validity checker
        EXPECT_FALSE(kvikio::S3EndpointWithPresignedUrl::is_url_valid(url));

        // Test explicit endpoint type specification
        {
          EXPECT_ANY_THROW({
            auto remote_handle = kvikio::RemoteHandle::open(
              url, kvikio::RemoteEndpointType::S3_PRESIGNED_URL, std::nullopt, 1);
          });
        }
      }
    }
  }

  // Invalid URLs
  {
    std::vector<std::string> urls{
      // Presigned URL should not use S3 scheme
      "s3://bucket-name/object-key-name",

      // Completely missing query
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name",

      // Missing key parameters ("X-Amz-..."") in query
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name?k0=v0&k1=v2"};
    for (auto const& url : urls) {
      EXPECT_FALSE(kvikio::S3EndpointWithPresignedUrl::is_url_valid(url));
    }
  }
}

TEST_F(RemoteHandleTest, check_webhdfs_url)
{
  {
    for (auto const& [url, expected_endpoint_type] : _urls_and_endpoint_type) {
      if (expected_endpoint_type == kvikio::RemoteEndpointType::WEBHDFS) {
        // Given that the URL is a valid WebHDFS URL

        // Test WebHDFS URL validity checker
        EXPECT_TRUE(kvikio::WebHdfsEndpoint::is_url_valid(url));

        // Test unified interface
        {
          auto remote_handle =
            kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1);
          EXPECT_EQ(remote_handle.type(), kvikio::RemoteEndpointType::WEBHDFS);
        }

        // Test explicit endpoint type specification
        {
          EXPECT_NO_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::WEBHDFS, std::nullopt, 1);
          });
        }
      } else {
        // Given that the URL is not an WebHDFS URL

        // Test WebHDFS URL validity checker
        EXPECT_FALSE(kvikio::WebHdfsEndpoint::is_url_valid(url));

        // Test explicit endpoint type specification
        {
          EXPECT_ANY_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::WEBHDFS, std::nullopt, 1);
          });
        }
      }
    }
  }

  // Invalid URLs
  {
    std::vector<std::string> urls{// Missing file
                                  "https://host:1234/webhdfs/v1",
                                  "https://host:1234/webhdfs/v1/",

                                  // Missing WebHDFS identifier
                                  "https://host:1234/data.bin",

                                  // Missing port number
                                  "https://host/webhdfs/v1/data.bin"};
    for (auto const& url : urls) {
      EXPECT_FALSE(kvikio::WebHdfsEndpoint::is_url_valid(url));
    }
  }
}
