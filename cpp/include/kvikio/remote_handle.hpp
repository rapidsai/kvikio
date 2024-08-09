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
#include <memory>
#include <stdexcept>
#include <string>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>

#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

#include <chrono>
using namespace std::chrono;

namespace kvikio {
namespace detail {

/**
 * Stream implementation of a fixed size buffer
 */
class BufferAsStream : public Aws::IOStream {
 public:
  using Base = Aws::IOStream;
  explicit BufferAsStream(std::streambuf* buf) : Base(buf) {}

  ~BufferAsStream() override = default;
};

class S3Context {
 public:
  S3Context() : _client{S3Context::create_client()} {}

  Aws::S3::S3Client& client() { return *_client; }

  static S3Context& default_context()
  {
    static S3Context _default_context;
    return _default_context;
  }

  S3Context(S3Context const&)      = delete;
  void operator=(S3Context const&) = delete;

 private:
  static void ensure_aws_s3_api_init()
  {
    static bool not_initalized{true};
    if (not_initalized) {
      std::cout << "ensure_aws_s3_api_initalized INIT" << std::endl;
      not_initalized = false;

      Aws::SDKOptions options;
      // options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Error;
      Aws::InitAPI(options);  // Should only be called once.
    }
  }

  static std::shared_ptr<Aws::S3::S3Client> create_client()
  {
    S3Context::ensure_aws_s3_api_init();

    Aws::Client::ClientConfiguration clientConfig;
    // Optional: Set to the AWS Region (overrides config file).
    // clientConfig.region = "us-east-1";

    const char* endpointOverride = getenv("AWS_ENDPOINT_URL");
    if (endpointOverride != nullptr) { clientConfig.endpointOverride = endpointOverride; }

    // You don't normally have to test that you are authenticated. But the S3 service permits
    // anonymous requests, thus the s3Client will return "success" even if you are
    // unauthenticated, which can be confusing to a new user.
    auto provider = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>("alloc-tag");
    auto creds    = provider->GetAWSCredentials();
    if (creds.IsEmpty()) {
      throw std::runtime_error(std::string("Failed authentication to ") + endpointOverride);
    }
    auto ret = std::make_shared<Aws::S3::S3Client>(Aws::S3::S3Client(clientConfig));

    // Try the connection
    auto outcome = ret->ListBuckets();
    if (!outcome.IsSuccess()) {
      throw std::runtime_error(std::string("S3 error: ") + outcome.GetError().GetMessage());
    }
    return ret;
  }

  std::shared_ptr<Aws::S3::S3Client> _client;
};

inline std::size_t get_s3_file_size(const std::string& bucket_name, const std::string& object_name)
{
  KVIKIO_NVTX_FUNC_RANGE();
  Aws::S3::Model::HeadObjectRequest req;
  req.SetBucket(bucket_name.c_str());
  req.SetKey(object_name.c_str());
  Aws::S3::Model::HeadObjectOutcome outcome = S3Context::default_context().client().HeadObject(req);
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
    KVIKIO_NVTX_FUNC_RANGE("AWS S3 receive", size);
    auto t0 = high_resolution_clock::now();

    auto& default_context = detail::S3Context::default_context();
    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(_bucket_name.c_str());
    req.SetKey(_object_name.c_str());
    const std::string byte_range =
      "bytes=" + std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
    req.SetRange(byte_range.c_str());

    // To write directly to `buf`, we register a "factory" that wraps a buffer as a output stream.
    Aws::Utils::Stream::PreallocatedStreamBuf buf_stream(static_cast<unsigned char*>(buf), size);
    req.SetResponseStreamFactory(
      [&]() { return Aws::New<detail::BufferAsStream>("BufferAsStream", &buf_stream); });

    Aws::S3::Model::GetObjectOutcome outcome = default_context.client().GetObject(req);
    if (!outcome.IsSuccess()) {
      const Aws::S3::S3Error& err = outcome.GetError();
      throw std::runtime_error(err.GetExceptionName() + ": " + err.GetMessage());
    }
    const std::size_t n = outcome.GetResult().GetContentLength();
    if (n != size) {
      throw std::runtime_error("S3 read of " + std::to_string(size) + " bytes failed, received " +
                               std::to_string(n) + " bytes");
    }
    auto t1        = high_resolution_clock::now();
    float duration = size / (duration_cast<microseconds>(t1 - t0).count() / 1000000.0);

    std::cout << "RemoteHandle::read_to_host() - buf: " << buf << ", size: " << size
              << ", file_offset: " << file_offset << ", bw: " << duration / (2 << 20) << " MiB/s"
              << std::endl;
    return n;
  }

  std::size_t read(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::read()", size);
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
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::pread()", size);
    std::cout << "RemoteHandle::pread()" << std::endl;
    auto task = [this](void* devPtr_base,
                       std::size_t size,
                       std::size_t file_offset,
                       std::size_t devPtr_offset) -> std::size_t {
      return read(static_cast<char*>(devPtr_base) + devPtr_offset, size, file_offset);
    };
    return parallel_io(task, buf, size, file_offset, posix_bounce_buffer_size, 0);
  }
};

}  // namespace kvikio
