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

#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace detail {

inline Aws::S3::S3Client _get_s3_client()  // TODO: spilt into a InitAPI and a S3Client function
{
  Aws::SDKOptions options;
  // options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Error;
  Aws::InitAPI(options);  // Should only be called once.

  Aws::Client::ClientConfiguration clientConfig;
  // Optional: Set to the AWS Region (overrides config file).
  // clientConfig.region = "us-east-1";

  const char* endpointOverride = getenv("AWS_ENDPOINT_URL");
  if (endpointOverride != nullptr) { clientConfig.endpointOverride = endpointOverride; }

  // You don't normally have to test that you are authenticated. But the S3 service permits
  // anonymous requests, thus the s3Client will return "success" and 0 buckets even if you are
  // unauthenticated, which can be confusing to a new user.
  auto provider = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>("alloc-tag");
  auto creds    = provider->GetAWSCredentials();
  if (creds.IsEmpty()) {
    std::cerr << "Failed authentication to " << endpointOverride << std::endl;
  }

  auto ret = Aws::S3::S3Client(clientConfig);

  // Try to use the connection
  auto outcome = ret.ListBuckets();
  if (!outcome.IsSuccess()) {
    std::cerr << "Failed with error: " << outcome.GetError() << std::endl;
  }

  // TODO: call Aws::ShutdownAPI(options) on exit?
  return ret;
}

inline const Aws::S3::S3Client& get_s3_client()
{
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

inline std::pair<std::string, std::string> parse_s3_path(const std::string& path)
{
  if (path.empty()) { throw std::invalid_argument("The remote path cannot be an empty string."); }
  if (path.size() < 5 || path.substr(0, 5) != "s3://") {
    throw std::invalid_argument("The remote path must start with the S3 scheme (\"s3://\").");
  }
  std::string p = path.substr(5);
  if (p.empty()) { throw std::invalid_argument("The remote path cannot be an empty string."); }
  size_t pos = p.find_first_of('/');
  if (pos == 0) { throw std::invalid_argument("The remote path does not contain a bucket name."); }
  return std::make_pair(p.substr(0, pos), (pos == std::string::npos) ? "" : p.substr(pos + 1));
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

  RemoteHandle(const std::string& remote_path)
  {
    auto [bucket_name, object_name] = detail::parse_s3_path(remote_path);
    _bucket_name                    = std::move(bucket_name);
    _object_name                    = std::move(object_name);
    _nbytes                         = detail::get_s3_file_size(_bucket_name, _object_name);

    std::cout << "RemoteHandle() - remote_path: " << remote_path
              << ", bucket_name: " << _bucket_name << ", object_name: " << _object_name
              << ", nbytes: " << _nbytes << std::endl;
  }

  /**
   * @brief Get the file size
   *
   * @return The number of bytes
   */
  [[nodiscard]] inline std::size_t nbytes() const { return _nbytes; }

  std::size_t read_to_host(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    std::cout << "RemoteHandle::read_to_host() - buf: " << buf << ", size: " << size
              << ", file_offset: " << file_offset << std::endl;

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

  std::size_t read(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    if (is_host_memory(buf)) { return read_to_host(buf, size, file_offset); }

    CUcontext ctx = get_context_from_pointer(buf);
    PushAndPopContext c(ctx);

    auto alloc         = detail::AllocRetain::instance().get();  // Host memory allocation
    CUdeviceptr devPtr = convert_void2deviceptr(buf);
    CUstream stream    = detail::StreamsByThread::get();

    std::size_t cur_file_offset = convert_size2off(file_offset);
    std::size_t byte_remaining  = convert_size2off(size);

    while (byte_remaining > 0) {
      const std::size_t nbytes_requested = std::min(posix_bounce_buffer_size, byte_remaining);
      std::size_t nbytes_got             = nbytes_requested;
      nbytes_got = read_to_host(alloc.get(), nbytes_requested, cur_file_offset);
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(devPtr, alloc.get(), nbytes_got, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
      cur_file_offset += nbytes_got;
      devPtr += nbytes_got;
      byte_remaining -= nbytes_got;
    }
    return size;
  }

  std::future<std::size_t> pread(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    // Notice, by passing `this`, `std::async`, binds `RemoteHandle::read` to `this`
    // automatically.
    return std::async(std::launch::deferred, &RemoteHandle::read, this, buf, size, file_offset);
  }
};

}  // namespace kvikio
