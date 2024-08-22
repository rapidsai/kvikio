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
#include <utility>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>

#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace detail {

/**
 * Stream implementation of a fixed size buffer.
 */
class BufferAsStream : public Aws::IOStream {
 public:
  using Base = Aws::IOStream;
  explicit BufferAsStream(std::streambuf* buf) : Base(buf) {}

  ~BufferAsStream() override = default;
};

/**
 * @brief Given a file path like "s3://<bucket>/<object>", return the name of the bucket and object.
 *
 * @param path S3 file path.
 * @return Pair of strings: [bucket-name, object-name].
 */
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
 * @brief S3 context, which initializes and maintains the S3 SDK and client.
 *
 * If not given an existing S3 client, S3Context calls `Aws::InitAPI()` and `Aws::ShutdownAPI`,
 * which inherit some limitations from the SDK.
 *  - The SDK for C++ and its dependencies use C++ static objects, and the order of static object
 *    destruction is not determined by the C++ standard. To avoid memory issues caused by the
 *    nondeterministic order of static variable destruction, do not wrap `S3Context` in another
 *    static object.
 *  - Please construct and destruct `S3Context` from the same thread (use a dedicated thread if
 *    necessary). This avoids problems in initializing the dependent Common RunTime C libraries.
 */
class S3Context {
 private:
  // We use a shared point since constructing a default `Aws::S3::S3Client` before calling
  // `Aws::InitAPI` is illegal.
  std::shared_ptr<Aws::S3::S3Client> _client;
  // Only call `Aws::ShutdownAPI`, if `Aws::InitAPI` was called on construction.
  const bool _shutdown_s3_api;

 public:
  /**
   * @brief Create a context given an existing S3 client
   *
   * The S3 SDK isn't initialized.
   *
   * @param client The S3 client
   */
  S3Context(std::shared_ptr<Aws::S3::S3Client> client)
    : _client{std::move(client)}, _shutdown_s3_api{false}
  {
    if (!_client) { throw std::invalid_argument("S3Context(): S3 client cannot be null"); }
  }

  /**
   * @brief Create a new context with a newly created S3 client.
   *
   * The S3 SDK is automatically initialized on construction and shutdown on destruction.
   *
   * The new S3 client use the default `Aws::Client::ClientConfiguration`, thus please make sure
   * that AWS credentials have been configure on the system. A common way to do this, is to set the
   * environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
   *
   * Other relevant options are `AWS_DEFAULT_REGION` and `AWS_ENDPOINT_URL`, see
   * <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>.
   *
   * @param endpoint_override If not empty, the address of the S3 server. Takes precedences
   * over the `AWS_ENDPOINT_URL` environment variable.
   */
  S3Context(const std::string& endpoint_override = "") : _shutdown_s3_api{true}
  {
    // NB: `Aws::InitAPI` has to be called before everything in the SDK beside `Aws::SDKOptions`,
    // even before config structs like `Aws::Client::ClientConfiguration`.
    // Notice, we may call `Aws::InitAPI`, which is allowed see:
    // <https://github.com/aws/aws-sdk-cpp/blob/main/src/aws-cpp-sdk-core/source/Aws.cpp#L32>
    Aws::SDKOptions options;
    Aws::InitAPI(options);

    // Create a client config where `endpoint_override` takes precedences over `AWS_ENDPOINT_URL`
    Aws::Client::ClientConfiguration config;
    const char* ep = std::getenv("AWS_ENDPOINT_URL");
    if (!endpoint_override.empty()) {
      config.endpointOverride = endpoint_override;
    } else if (ep != nullptr && !std::string(ep).empty()) {
      config.endpointOverride = ep;
    }

    // We check authentication here to trigger an early exception.
    Aws::Auth::DefaultAWSCredentialsProviderChain provider;
    if (provider.GetAWSCredentials().IsEmpty()) {
      throw std::runtime_error("failed authentication to S3 server");
    }
    _client = std::make_shared<Aws::S3::S3Client>(config);
  }

  ~S3Context() noexcept
  {
    if (_shutdown_s3_api) {
      try {
        Aws::SDKOptions options;
        Aws::ShutdownAPI(options);
      } catch (const std::exception& e) {
        std::cerr << "~S3Context(): " << e.what() << std::endl;
      }
    }
  }

  /**
   * @brief Get a reference to the S3 client.
   *
   * @return S3 client.
   */
  Aws::S3::S3Client& client() { return *_client; }

  // No copy and move semantic
  S3Context(S3Context const&)       = delete;
  void operator=(S3Context const&)  = delete;
  S3Context(S3Context const&&)      = delete;
  void operator=(S3Context const&&) = delete;

  /**
   * @brief Get the size of a S3 file
   *
   * @param bucket_name The bucket name.
   * @param object_name The object name.
   * @return Size of the file in bytes.
   */
  std::size_t get_file_size(const std::string& bucket_name, const std::string& object_name)
  {
    KVIKIO_NVTX_FUNC_RANGE();
    Aws::S3::Model::HeadObjectRequest req;
    req.SetBucket(bucket_name.c_str());
    req.SetKey(object_name.c_str());
    Aws::S3::Model::HeadObjectOutcome outcome = client().HeadObject(req);
    if (!outcome.IsSuccess()) {
      const Aws::S3::S3Error& err = outcome.GetError();
      throw std::invalid_argument("get_file_size(): " + err.GetExceptionName() + ": " +
                                  err.GetMessage());
    }
    return outcome.GetResult().GetContentLength();
  }
};

/**
 * @brief Handle of remote file (currently, only AWS S3 is supported).
 */
class RemoteHandle {
 private:
  std::string _bucket_name{};
  std::string _object_name{};
  std::size_t _nbytes{};
  std::shared_ptr<S3Context> _context;

 public:
  // Use of a default constructed instance is undefined behavior.
  RemoteHandle() noexcept = default;

  /**
   * @brief Construct from a bucket and object name pair.
   *
   * @param context The S3 context used for the connection to the remove server.
   * @param bucket_and_object_name Name pair <bucket, object>.
   */
  RemoteHandle(std::shared_ptr<S3Context> context,
               std::pair<std::string, std::string> bucket_and_object_name)
  {
    if (!context) { throw std::invalid_argument("RemoteHandle(): context cannot be null"); }
    _context     = std::move(context);
    _bucket_name = std::move(bucket_and_object_name.first);
    _object_name = std::move(bucket_and_object_name.second);
    _nbytes      = _context->get_file_size(_bucket_name, _object_name);
  }

  /**
   * @brief Construct from a bucket and object name.
   *
   * @param context The S3 context used for the connection to the remove server.
   * @param bucket_name Name of the bucket.
   * @param object_name Name of the object.
   */
  RemoteHandle(std::shared_ptr<S3Context> context, std::string bucket_name, std::string object_name)
    : RemoteHandle(std::move(context),
                   std::make_pair(std::move(bucket_name), std::move(object_name)))
  {
  }

  /**
   * @brief Construct from a remote path such as "s3://<bucket>/<object>".
   *
   * @param context The S3 context used for the connection to the remove server.
   * @param remote_path Remote file path.
   */
  RemoteHandle(std::shared_ptr<S3Context> context, const std::string& remote_path)
    : RemoteHandle(std::move(context), detail::parse_s3_path(remote_path))
  {
  }

  /**
   * @brief Get the file size.
   *
   * Note, this is very fast, no communication needed.
   *
   * @return The number of bytes.
   */
  [[nodiscard]] std::size_t nbytes() const { return _nbytes; }

  /**
   * @brief Read from remote source into host memory.
   *
   * @param buf Pointer to host memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @return Number of bytes read.
   */
  std::size_t read_to_host(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    KVIKIO_NVTX_FUNC_RANGE("AWS S3 receive", size);

    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(_bucket_name.c_str());
    req.SetKey(_object_name.c_str());
    const std::string byte_range =
      "bytes=" + std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
    req.SetRange(byte_range.c_str());

    // To write directly to `buf`, we register a "factory" that wraps a buffer as an output stream.
    // Notice, the AWS SDK will handle the freeing of the returned `detail::BufferAsStream`:
    // <https://github.com/aws/aws-sdk-cpp/blob/main/src/aws-cpp-sdk-core/source/utils/stream/ResponseStream.cpp#L78>
    Aws::Utils::Stream::PreallocatedStreamBuf buf_stream(static_cast<unsigned char*>(buf), size);
    req.SetResponseStreamFactory(
      [&]() { return Aws::New<detail::BufferAsStream>("BufferAsStream", &buf_stream); });

    Aws::S3::Model::GetObjectOutcome outcome = _context->client().GetObject(req);
    if (!outcome.IsSuccess()) {
      const Aws::S3::S3Error& err = outcome.GetError();
      throw std::runtime_error(err.GetExceptionName() + ": " + err.GetMessage());
    }
    const std::size_t n = outcome.GetResult().GetContentLength();
    if (n != size) {
      throw std::runtime_error("S3 read of " + std::to_string(size) + " bytes failed, received " +
                               std::to_string(n) + " bytes");
    }
    return n;
  }

  /**
   * @brief Read from remote source into buffer (host or device memory).
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @return Number of bytes read, which is `size` always.
   */
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
      std::size_t nbytes_got = read_to_host(alloc.get(), nbytes_requested, cur_file_offset);
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(devPtr, alloc.get(), nbytes_got, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
      cur_file_offset += nbytes_got;
      devPtr += nbytes_got;
      byte_remaining -= nbytes_got;
    }
    return size;
  }

  /**
   * @brief Read from remote source into buffer (host or device memory) in parallel.
   *
   * Contrary to `FileHandle::pread()`, a task size of 16 MiB is used always.
   * See `kvikio::posix_bounce_buffer_size`.
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @return Number of bytes read, which is `size` always.
   */
  std::future<std::size_t> pread(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::pread()", size);
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
