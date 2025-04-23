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

#include <gtest/gtest.h>
#include <kvikio/remote_handle.hpp>

#include <unordered_map>
#include "utils/env.hpp"

TEST(RemoteHandleTest, s3_endpoint_constructor)
{
  std::unordered_map<std::string, std::string> env_var_entries{
    {"AWS_DEFAULT_REGION", "my_aws_default_region"},
    {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
    {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"},
    {"AWS_ENDPOINT_URL", "https://my_aws_endpoint_url"}};

  kvikio::test::EnvVarContext env_var_ctx{env_var_entries};
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
