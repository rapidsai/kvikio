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

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/remote_handle.hpp>

#include "utils/env.hpp"

TEST(RemoteHandleTest, s3_endpoint_constructor)
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

TEST(RemoteHandleTest, check_http)
{
  {
    std::vector<std::string> urls{
      "http://example.com", "https://example.com", "HTTP://example.com", "hTTpS://example.com"};
    for (auto const& url : urls) {
      EXPECT_TRUE(kvikio::HttpEndpoint::is_url_compatible(url));
    }
  }

  {
    std::vector<std::string> urls{"s3://example.com", "hdfs://example.com"};
    for (auto const& url : urls) {
      EXPECT_FALSE(kvikio::HttpEndpoint::is_url_compatible(url));
    }
  }
}

TEST(RemoteHandleTest, check_s3) {}

TEST(RemoteHandleTest, check_s3_with_presigned_url)
{
  {
    std::vector<std::string> urls{
      "https://bucket-name.s3.region-code.amazonaws.com/"
      "object-key-name?X-Amz-Algorithm=algo&X-Amz-Signature=sig"};
    for (auto const& url : urls) {
      EXPECT_TRUE(kvikio::S3EndpointWithPresignedUrl::is_url_compatible(url));
    }
  }

  {
    std::vector<std::string> urls{
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name"};
    for (auto const& url : urls) {
      EXPECT_FALSE(kvikio::S3EndpointWithPresignedUrl::is_url_compatible(url));
    }
  }
}

TEST(RemoteHandleTest, check_webhdfs) {}

TEST(RemoteHandleTest, unified_remote_file_constructor)
{
  //   std::size_t const num_bytes{1};

  //   {
  //     kvikio::test::EnvVarContext env_var_ctx{{"AWS_DEFAULT_REGION", "my_aws_default_region"},
  //                                             {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
  //                                             {"AWS_SECRET_ACCESS_KEY",
  //                                             "my_aws_secrete_access_key"},
  //                                             {"AWS_ENDPOINT_URL",
  //                                             "https://my_aws_endpoint_url"}};

  //     std::vector<std::pair<std::string, kvikio::RemoteFileType>> questions_and_answers{
  //       {"s3://geralt-of-rivia/bestiary.txt", kvikio::RemoteFileType::S3},  // AWS S3 URI

  //       {"https://school-of-the-wolf.s3.kaer-morhen-1.amazonaws.com/geralt-of-rivia/bestiary.txt",
  //        kvikio::RemoteFileType::S3},  // AWS S3 virtual-hosted-style URL

  //       {"https://s3.kaer-morhen-1.amazonaws.com/school-of-the-wolf/geralt-of-rivia/bestiary.txt",
  //        kvikio::RemoteFileType::S3},  // AWS S3 path-style URL (deprecated)

  //       {"https://school-of-the-wolf.s3.amazonaws.com/geralt-of-rivia/bestiary.txt",
  //        kvikio::RemoteFileType::S3},  // AWS S3 legacy global endpoint URL
  //     };

  //     for (auto const& [url, expected_type] : questions_and_answers) {
  //       auto remote_handle = kvikio::RemoteHandle::open(url, kvikio::RemoteFileType::AUTO,
  //       num_bytes); EXPECT_EQ(remote_handle.type(), expected_type);
  //     }
  //   }

  //   // Artificial URL and expected type of the remote file
  //   std::vector<std::pair<std::string, kvikio::RemoteFileType>> questions_and_answers{
  //     {"s3://geralt-of-rivia/bestiary.txt", kvikio::RemoteFileType::S3},  // AWS S3 URI

  //     {"https://school-of-the-wolf.s3.kaer-morhen-1.amazonaws.com/geralt-of-rivia/bestiary.txt",
  //      kvikio::RemoteFileType::S3},  // AWS S3 virtual-hosted-style URL

  //     {"https://s3.kaer-morhen-1.amazonaws.com/school-of-the-wolf/geralt-of-rivia/bestiary.txt",
  //      kvikio::RemoteFileType::S3},  // AWS S3 path-style URL (deprecated)

  //     {"https://school-of-the-wolf.s3.amazonaws.com/geralt-of-rivia/bestiary.txt",
  //      kvikio::RemoteFileType::S3},  // AWS S3 legacy global endpoint URL

  //     {"https://school-of-the-wol.s3.kaer-morhen-1.amazonaws.com/geralt-of-rivia/"
  //      "bestiary.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=protagonist",
  //      kvikio::RemoteFileType::S3_PRESIGNED_URL},

  //     {"http://skellige-isles.com:1234/webhdfs/v1/home/yennefer-of-vengerberg/"
  //      "how-to-subdue-djinn?op=OPEN",
  //      kvikio::RemoteFileType::WEBHDFS},

  //     {"http://novigrad.com/dandelion/saga_of_cirilla.txt", kvikio::RemoteFileType::HTTP},
  //   };

  //   for (auto const& [url, expected_type] : questions_and_answers) {
  //     auto remote_handle = kvikio::RemoteHandle::open(url);  // RemoteFileType::AUTO
  //     EXPECT_EQ(remote_handle.type(), expected_type);
  //   }
}
