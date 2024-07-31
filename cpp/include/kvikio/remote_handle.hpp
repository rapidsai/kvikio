/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>

namespace kvikio {
namespace detail {

inline Aws::S3::S3Client _get_s3_client()  // TODO: spilt into a InitAPI and a S3Client function
{
  std::cout << "_get_s3_client()" << std::endl;
  Aws::SDKOptions options;
  // Optionally change the log level for debugging.
  //   options.loggingOptions.logLevel = Utils::Logging::LogLevel::Debug;
  Aws::InitAPI(options);  // Should only be called once.

  Aws::Client::ClientConfiguration clientConfig;
  // Optional: Set to the AWS Region (overrides config file).
  // clientConfig.region = "us-east-1";

  const char* endpointOverride = getenv("AWS_ENDPOINT_URL");
  if (endpointOverride != nullptr) { clientConfig.endpointOverride = endpointOverride; }
  std::cout << "endpointOverride: " << endpointOverride << std::endl;

  // You don't normally have to test that you are authenticated. But the S3 service permits
  // anonymous requests, thus the s3Client will return "success" and 0 buckets even if you are
  // unauthenticated, which can be confusing to a new user.
  auto provider = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>("alloc-tag");
  auto creds    = provider->GetAWSCredentials();
  if (creds.IsEmpty()) { std::cerr << "Failed authentication" << std::endl; }

  auto ret     = Aws::S3::S3Client(clientConfig);
  auto outcome = ret.ListBuckets();

  if (!outcome.IsSuccess()) {
    std::cerr << "Failed with error: " << outcome.GetError() << std::endl;
  } else {
    std::cout << "Found " << outcome.GetResult().GetBuckets().size() << " buckets\n";
    for (auto& bucket : outcome.GetResult().GetBuckets()) {
      std::cout << bucket.GetName() << std::endl;
    }
  }

  // TODO: call Aws::ShutdownAPI(options) on exit?
  return ret;
}

inline const Aws::S3::S3Client& get_s3_client()
{
  std::cout << "get_s3_client()" << std::endl;
  static Aws::S3::S3Client ret = _get_s3_client();
  return ret;
}

inline std::size_t get_s3_file_size(const std::string& bucket_name, const std::string& object_name)
{
  Aws::S3::Model::HeadObjectRequest req;
  req.SetBucket(bucket_name.c_str());
  req.SetKey(object_name.c_str());
  Aws::S3::Model::HeadObjectOutcome outcome = get_s3_client().HeadObject(req);

  if (!outcome.IsSuccess()) {
    const Aws::S3::S3Error& err = outcome.GetError();
    throw std::invalid_argument("get_s3_file_size(): " + err.GetExceptionName() + ": " +
                                err.GetMessage());
  }
  return outcome.GetResult().GetContentLength();
}

}  // namespace detail

/**
 * @brief Handle of
 *
 * At the moment, only AWS S3 is the supported
 */
class RemoteHandle {
 private:
  std::string _bucket_name{};
  std::string _object_name{};
  std::size_t _nbytes{};

 public:
  RemoteHandle() noexcept = default;

  RemoteHandle(std::string bucket_name, std::string object_name)
    : _bucket_name(std::move(bucket_name)),
      _object_name(std::move(object_name)),
      _nbytes(detail::get_s3_file_size(_bucket_name, _object_name))
  {
    std::cout << "RemoteHandle() - bucket_name: " << _bucket_name
              << ", object_name: " << _object_name << ", nbytes: " << _nbytes << std::endl;
  }

  /**
   * @brief Get the file size
   *
   * @return The number of bytes
   */
  [[nodiscard]] inline std::size_t nbytes() const { return _nbytes; }

  std::size_t read(void* buf,
                   std::size_t size,
                   std::size_t file_offset = 0,
                   std::size_t task_size   = defaults::task_size())
  {
    std::cout << "RemoteHandle::read() - buf: " << buf << ", size: " << size
              << ", file_offset: " << file_offset << ", task_size: " << task_size << std::endl;

    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(_bucket_name.c_str());
    req.SetKey(_object_name.c_str());
    const std::string byte_range =
      "bytes=" + std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
    req.SetRange(byte_range.c_str());

    // TODO: use a custom factory that writes directly to `buf`
    //       see <https://github.com/apache/arrow/pull/7098>
    req.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>("KvikIOAllocationTag"); });

    Aws::S3::Model::GetObjectOutcome outcome = detail::get_s3_client().GetObject(req);
    if (!outcome.IsSuccess()) {
      const Aws::S3::S3Error& err = outcome.GetError();
      throw std::runtime_error(err.GetExceptionName() + ": " + err.GetMessage());
    }
    const std::size_t n = outcome.GetResult().GetContentLength();
    if (n != size) {
      throw std::runtime_error("S3 read of " + std::to_string(size) + " bytes failed, received " +
                               std::to_string(n) + " bytes");
    }
    outcome.GetResult().GetBody().read(static_cast<char*>(buf), size);
    return n;
  }
};

}  // namespace kvikio
