/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

namespace {

/**
 * @brief Bounce buffer in pinned host memory.
 *
 * @note Is not thread-safe.
 */
class BounceBufferH2D {
  CUstream _stream;                 // The CUDA stream to use.
  CUdeviceptr _dev;                 // The output device buffer.
  AllocRetain::Alloc _host_buffer;  // The host buffer to bounce data on.
  std::ptrdiff_t _dev_offset{0};    // Number of bytes written to `_dev`.
  std::ptrdiff_t _host_offset{0};   // Number of bytes written to `_host` (resets on flush).

 public:
  /**
   * @brief Create a bounce buffer for an output device buffer.
   *
   * @param stream The CUDA stream used throughout the lifetime of the bounce buffer.
   * @param device_buffer The output device buffer (final destination of the data).
   */
  BounceBufferH2D(CUstream stream, void* device_buffer)
    : _stream{stream},
      _dev{convert_void2deviceptr(device_buffer)},
      _host_buffer{AllocRetain::instance().get()}
  {
  }

  /**
   * @brief The bounce buffer if flushed to device on destruction.
   */
  ~BounceBufferH2D() noexcept
  {
    try {
      flush();
    } catch (CUfileException const& e) {
      std::cerr << "BounceBufferH2D error on final flush: ";
      std::cerr << e.what();
      std::cerr << std::endl;
    }
  }

 private:
  /**
   * @brief Write host memory to the output device buffer.
   *
   * @param src The host memory source.
   * @param size Number of bytes to write.
   */
  void write_to_device(void const* src, std::size_t size)
  {
    if (size > 0) {
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(_dev + _dev_offset, src, size, _stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream));
      _dev_offset += size;
    }
  }

  /**
   * @brief Flush the bounce buffer by writing everything to the output device buffer.
   */
  void flush()
  {
    write_to_device(_host_buffer.get(), _host_offset);
    _host_offset = 0;
  }

 public:
  /**
   * @brief Write host memory to the bounce buffer (also host memory).
   *
   * Only when the bounce buffer has been filled up is data copied to the output device buffer.
   *
   * @param data The host memory source.
   * @param size Number of bytes to write.
   */
  void write(char const* data, std::size_t size)
  {
    if (_host_buffer.size() - _host_offset < size) {  // Not enough space left in the bounce buffer
      flush();
      assert(_host_offset == 0);
    }
    if (_host_buffer.size() < size) {
      // If still not enough space, we just copy the data to the device. This only happens when
      // `defaults::bounce_buffer_size()` is smaller than 16kb thus no need to performance
      // optimize for this case.
      write_to_device(data, size);
    } else if (size > 0) {
      std::memcpy(_host_buffer.get(_host_offset), data, size);
      _host_offset += size;
    }
  }
};

}  // namespace

HttpEndpoint::HttpEndpoint(std::string url) : _url{std::move(url)} {}

std::string HttpEndpoint::str() const { return _url; }

void HttpEndpoint::setopt(CurlHandle& curl) { curl.setopt(CURLOPT_URL, _url.c_str()); }

void S3Endpoint::setopt(CurlHandle& curl)
{
  curl.setopt(CURLOPT_URL, _url.c_str());
  curl.setopt(CURLOPT_AWS_SIGV4, _aws_sigv4.c_str());
  curl.setopt(CURLOPT_USERPWD, _aws_userpwd.c_str());
}

std::string S3Endpoint::unwrap_or_default(std::optional<std::string> aws_arg,
                                          std::string const& env_var,
                                          std::string const& err_msg)
{
  if (aws_arg.has_value()) { return std::move(*aws_arg); }

  char const* env = std::getenv(env_var.c_str());
  if (env == nullptr) {
    if (err_msg.empty()) { return std::string(); }
    throw std::invalid_argument(err_msg);
  }
  return std::string(env);
}

std::string S3Endpoint::url_from_bucket_and_object(std::string const& bucket_name,
                                                   std::string const& object_name,
                                                   std::optional<std::string> const& aws_region,
                                                   std::optional<std::string> aws_endpoint_url)
{
  auto const endpoint_url = unwrap_or_default(std::move(aws_endpoint_url), "AWS_ENDPOINT_URL");
  std::stringstream ss;
  if (endpoint_url.empty()) {
    auto const region =
      unwrap_or_default(std::move(aws_region),
                        "AWS_DEFAULT_REGION",
                        "S3: must provide `aws_region` if AWS_DEFAULT_REGION isn't set.");
    // We default to the official AWS url scheme.
    ss << "https://" << bucket_name << ".s3." << region << ".amazonaws.com/" << object_name;
  } else {
    ss << endpoint_url << "/" << bucket_name << "/" << object_name;
  }
  return ss.str();
}

std::pair<std::string, std::string> S3Endpoint::parse_s3_url(std::string const& s3_url)
{
  // Regular expression to match s3://<bucket>/<object>
  std::regex const pattern{R"(^s3://([^/]+)/(.+))", std::regex_constants::icase};
  std::smatch matches;
  if (std::regex_match(s3_url, matches, pattern)) { return {matches[1].str(), matches[2].str()}; }
  throw std::invalid_argument("Input string does not match the expected S3 URL format.");
}

S3Endpoint::S3Endpoint(std::string url,
                       std::optional<std::string> aws_region,
                       std::optional<std::string> aws_access_key,
                       std::optional<std::string> aws_secret_access_key)
  : _url{std::move(url)}
{
  // Regular expression to match http[s]://
  std::regex pattern{R"(^https?://.*)", std::regex_constants::icase};
  if (!std::regex_search(_url, pattern)) {
    throw std::invalid_argument("url must start with http:// or https://");
  }

  auto const region =
    unwrap_or_default(std::move(aws_region),
                      "AWS_DEFAULT_REGION",
                      "S3: must provide `aws_region` if AWS_DEFAULT_REGION isn't set.");

  auto const access_key =
    unwrap_or_default(std::move(aws_access_key),
                      "AWS_ACCESS_KEY_ID",
                      "S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID isn't set.");

  auto const secret_access_key = unwrap_or_default(
    std::move(aws_secret_access_key),
    "AWS_SECRET_ACCESS_KEY",
    "S3: must provide `aws_secret_access_key` if AWS_SECRET_ACCESS_KEY isn't set.");

  // Create the CURLOPT_AWS_SIGV4 option
  {
    std::stringstream ss;
    ss << "aws:amz:" << region << ":s3";
    _aws_sigv4 = ss.str();
  }
  // Create the CURLOPT_USERPWD option
  // Notice, curl uses `secret_access_key` to generate a AWS V4 signature. It is NOT included
  // in the http header. See
  // <https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html>
  {
    std::stringstream ss;
    ss << access_key << ":" << secret_access_key;
    _aws_userpwd = ss.str();
  }
}

S3Endpoint::S3Endpoint(std::string const& bucket_name,
                       std::string const& object_name,
                       std::optional<std::string> aws_region,
                       std::optional<std::string> aws_access_key,
                       std::optional<std::string> aws_secret_access_key,
                       std::optional<std::string> aws_endpoint_url)
  : S3Endpoint(
      url_from_bucket_and_object(bucket_name, object_name, aws_region, std::move(aws_endpoint_url)),
      std::move(aws_region),
      std::move(aws_access_key),
      std::move(aws_secret_access_key))
{
}

std::string S3Endpoint::str() const { return _url; }

RemoteHandle::RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint, std::size_t nbytes)
  : _endpoint{std::move(endpoint)}, _nbytes{nbytes}
{
}

RemoteHandle::RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint)
{
  auto curl = create_curl_handle();

  endpoint->setopt(curl);
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
  curl.perform();
  curl_off_t cl;
  curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
  if (cl < 0) {
    throw std::runtime_error("cannot get size of " + endpoint->str() +
                             ", content-length not provided by the server");
  }
  _nbytes   = cl;
  _endpoint = std::move(endpoint);
}

std::size_t RemoteHandle::nbytes() const noexcept { return _nbytes; }

RemoteEndpoint const& RemoteHandle::endpoint() const noexcept { return *_endpoint; }

namespace {

/**
 * @brief Context used by the "CURLOPT_WRITEFUNCTION" callbacks.
 */
struct CallbackContext {
  char* buf;              // Output buffer to read into.
  std::size_t size;       // Total number of bytes to read.
  std::ptrdiff_t offset;  // Offset into `buf` to start reading.
  bool overflow_error;    // Flag to indicate overflow.
  CallbackContext(void* buf, std::size_t size)
    : buf{static_cast<char*>(buf)}, size{size}, offset{0}, overflow_error{0}
  {
  }
  BounceBufferH2D* bounce_buffer{nullptr};  // Only used by callback_device_memory
};

/**
 * @brief A "CURLOPT_WRITEFUNCTION" to copy downloaded data to the output host buffer.
 *
 * See <https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html>.
 *
 * @param data Data downloaded by libcurl that is ready for consumption.
 * @param size Size of each element in `nmemb`; size is always 1.
 * @param nmemb Size of the data in `nmemb`.
 * @param context A pointer to an instance of `CallbackContext`.
 */
std::size_t callback_host_memory(char* data, std::size_t size, std::size_t nmemb, void* context)
{
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle - callback_host_memory()", nbytes);
  std::memcpy(ctx->buf + ctx->offset, data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}

/**
 * @brief A "CURLOPT_WRITEFUNCTION" to copy downloaded data to the output device buffer.
 *
 * See <https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html>.
 *
 * @param data Data downloaded by libcurl that is ready for consumption.
 * @param size Size of each element in `nmemb`; size is always 1.
 * @param nmemb Size of the data in `nmemb`.
 * @param context A pointer to an instance of `CallbackContext`.
 */
std::size_t callback_device_memory(char* data, std::size_t size, std::size_t nmemb, void* context)
{
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle - callback_device_memory()", nbytes);

  ctx->bounce_buffer->write(data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}
}  // namespace

std::size_t RemoteHandle::read(void* buf, std::size_t size, std::size_t file_offset)
{
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle::read()", size);

  if (file_offset + size > _nbytes) {
    std::stringstream ss;
    ss << "cannot read " << file_offset << "+" << size << " bytes into a " << _nbytes
       << " bytes file (" << _endpoint->str() << ")";
    throw std::invalid_argument(ss.str());
  }
  bool const is_host_mem = is_host_memory(buf);
  auto curl              = create_curl_handle();
  _endpoint->setopt(curl);

  std::string const byte_range =
    std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
  curl.setopt(CURLOPT_RANGE, byte_range.c_str());

  if (is_host_mem) {
    curl.setopt(CURLOPT_WRITEFUNCTION, callback_host_memory);
  } else {
    curl.setopt(CURLOPT_WRITEFUNCTION, callback_device_memory);
  }
  CallbackContext ctx{buf, size};
  curl.setopt(CURLOPT_WRITEDATA, &ctx);

  try {
    if (is_host_mem) {
      curl.perform();
    } else {
      PushAndPopContext c(get_context_from_pointer(buf));
      // We use a bounce buffer to avoid many small memory copies to device. Libcurl has a
      // maximum chunk size of 16kb (`CURL_MAX_WRITE_SIZE`) but chunks are often much smaller.
      BounceBufferH2D bounce_buffer(detail::StreamsByThread::get(), buf);
      ctx.bounce_buffer = &bounce_buffer;
      curl.perform();
    }
  } catch (std::runtime_error const& e) {
    if (ctx.overflow_error) {
      std::stringstream ss;
      ss << "maybe the server doesn't support file ranges? [" << e.what() << "]";
      throw std::overflow_error(ss.str());
    }
    throw;
  }
  return size;
}

std::future<std::size_t> RemoteHandle::pread(void* buf,
                                             std::size_t size,
                                             std::size_t file_offset,
                                             std::size_t task_size)
{
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle::pread()", size);
  auto task = [this](void* devPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t devPtr_offset) -> std::size_t {
    return read(static_cast<char*>(devPtr_base) + devPtr_offset, size, file_offset);
  };
  return parallel_io(task, buf, size, file_offset, task_size, 0);
}

}  // namespace kvikio
