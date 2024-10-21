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

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace detail {

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
inline std::size_t callback_host_memory(char* data,
                                        std::size_t size,
                                        std::size_t nmemb,
                                        void* context)
{
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE("RemoteHandle - callback_host_memory()", nbytes);
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
inline std::size_t callback_device_memory(char* data,
                                          std::size_t size,
                                          std::size_t nmemb,
                                          void* context)
{
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE("RemoteHandle - callback_device_memory()", nbytes);

  CUstream stream = detail::StreamsByThread::get();
  CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
    convert_void2deviceptr(ctx->buf + ctx->offset), data, nbytes, stream));
  // We have to sync since curl might overwrite or free `data`.
  CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));

  ctx->offset += nbytes;
  return nbytes;
}

}  // namespace detail

/**
 * @brief Abstract base class for remote endpoints.
 *
 * In this context, an endpoint refers to a remote file using a specific communication protocol.
 *
 * Each communication protocol, such as HTTP or S3, needs to implement this ABC and implement
 * its own ctor that takes communication protocol specific arguments.
 */
class RemoteEndpoint {
 public:
  /**
   * @brief Set needed connection options on a curl handle.
   *
   * Subsequently, a call to `curl.perform()` should connect to the endpoint.
   *
   * @param curl The curl handle.
   */
  virtual void setopt(CurlHandle& curl) = 0;

  /**
   * @brief Get a description of this remote point instance.
   *
   * @returns A string description.
   */
  virtual std::string str() = 0;

  virtual ~RemoteEndpoint() = default;
};

/**
 * @brief A remote endpoint using http.
 */
class HttpEndpoint : public RemoteEndpoint {
 private:
  std::string _url;

 public:
  /**
   * @brief Create an http endpoint from a url.
   *
   * @param url The full http url to the remote file.
   */
  HttpEndpoint(std::string url) : _url{std::move(url)} {}
  void setopt(CurlHandle& curl) override { curl.setopt(CURLOPT_URL, _url.c_str()); }
  std::string str() override { return _url; }
  ~HttpEndpoint() override = default;
};

/**
 * @brief A remote endpoint using AWS's S3 protocol.
 */
class S3Endpoint : public RemoteEndpoint {
 private:
  std::string _url;
  std::string _aws_sigv4;
  std::string _aws_userpwd;

  /**
   * @brief Unwrap an optional parameter, obtaining a default from the environment.
   *
   * If not nullopt, the optional's value is returned otherwise the environment
   * variable `env_var` is used. If that also doesn't have a value:
   *   - if `err_msg` is empty, the empty string is returned.
   *   - if `err_msg` is not empty, `std::invalid_argument(`err_msg`)` is thrown.
   *
   * @param value The value to unwrap.
   * @param env_var The name of the environment variable to check if `value` isn't set.
   * @param err_msg The error message to throw on error or the empty string.
   * @return The parsed AWS argument or the empty string.
   */
  static std::string parse_aws_argument(std::optional<std::string> aws_arg,
                                        std::string const& env_var,
                                        std::string const& err_msg = "")
  {
    if (aws_arg.has_value()) { return std::move(*aws_arg); }

    char const* env = std::getenv(env_var.c_str());
    if (env == nullptr) {
      if (err_msg.empty()) { return std::string(); }
      throw std::invalid_argument(err_msg);
    }
    return std::string(env);
  }

  /**
   * @brief Get url from a AWS S3 bucket and object name.
   *
   * @throws std::invalid_argument if no region is specified and no default region is
   * specified in the environment.
   *
   * @param bucket_name The name of the S3 bucket.
   * @param object_name The name of the S3 object.
   * @param aws_region The AWS region, such as "us-east-1", to use. If nullopt, the value of the
   * `AWS_DEFAULT_REGION` environment variable is used.
   * @param aws_endpoint_url Overwrite the endpoint url to use. If nullopt, the value of
   * the `AWS_ENDPOINT_URL` environment variable is used. If this is also not set, the regular
   * AWS url scheme is used: "https://<bucket>.s3.<region>.amazonaws.com/<object>"
   */
  static std::string url_from_bucket_and_object(std::string const& bucket_name,
                                                std::string const& object_name,
                                                std::optional<std::string> const& aws_region,
                                                std::optional<std::string> aws_endpoint_url)
  {
    auto const endpoint_url = parse_aws_argument(std::move(aws_endpoint_url), "AWS_ENDPOINT_URL");
    std::stringstream ss;
    if (endpoint_url.empty()) {
      auto const region =
        parse_aws_argument(std::move(aws_region),
                           "AWS_DEFAULT_REGION",
                           "S3: must provide `aws_region` if AWS_DEFAULT_REGION isn't set.");
      // We default to the official AWS url scheme.
      ss << "https://" << bucket_name << ".s3." << region << ".amazonaws.com/" << object_name;
    } else {
      ss << endpoint_url << "/" << bucket_name << "/" << object_name;
    }
    return ss.str();
  }

 public:
  /**
   * @brief Given an url like "s3://<bucket>/<object>", return the name of the bucket and object.
   *
   * @throws std::invalid_argument if url is ill-formed or is missing the bucket or object name.
   *
   * @param s3_url S3 url.
   * @return Pair of strings: [bucket-name, object-name].
   */
  [[nodiscard]] static std::pair<std::string, std::string> parse_s3_url(std::string const& s3_url)
  {
    // Regular expression to match s3://<bucket>/<object>
    std::regex pattern{R"(s3://([^/]+)/(.+))"};
    std::smatch matches;
    if (std::regex_match(s3_url, matches, pattern)) { return {matches[1].str(), matches[2].str()}; }
    throw std::invalid_argument("Input string does not match the expected S3 URL format.");
  }

  /**
   * @brief Create a S3 endpoint from a url.
   *
   * @param url The full http url to the S3 file. NB: this should be an url starting with
   * "http://" or "https://". If you have an S3 url of the form "s3://<bucket>/<object>",
   * please use `S3Endpoint::parse_s3_url()` to convert it.
   * @param aws_region The AWS region, such as "us-east-1", to use. If nullopt, the value of the
   * `AWS_DEFAULT_REGION` environment variable is used.
   * @param aws_access_key The AWS access key to use. If nullopt, the value of the
   * `AWS_ACCESS_KEY_ID` environment variable is used.
   * @param aws_secret_access_key The AWS secret access key to use. If nullopt, the value of the
   * `AWS_SECRET_ACCESS_KEY` environment variable is used.
   */
  S3Endpoint(std::string url,
             std::optional<std::string> aws_region            = std::nullopt,
             std::optional<std::string> aws_access_key        = std::nullopt,
             std::optional<std::string> aws_secret_access_key = std::nullopt)
    : _url{std::move(url)}
  {
    auto const region =
      parse_aws_argument(std::move(aws_region),
                         "AWS_DEFAULT_REGION",
                         "S3: must provide `aws_region` if AWS_DEFAULT_REGION isn't set.");

    auto const access_key =
      parse_aws_argument(std::move(aws_access_key),
                         "AWS_ACCESS_KEY_ID",
                         "S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID isn't set.");

    auto const secret_access_key = parse_aws_argument(
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
    {
      std::stringstream ss;
      ss << access_key << ":" << secret_access_key;
      _aws_userpwd = ss.str();
    }
  }

  /**
   * @brief Create a S3 endpoint from a bucket and object name.
   *
   * @param bucket_name The name of the S3 bucket.
   * @param object_name The name of the S3 object.
   * @param aws_region The AWS region, such as "us-east-1", to use. If nullopt, the value of the
   * `AWS_DEFAULT_REGION` environment variable is used.
   * @param aws_access_key The AWS access key to use. If nullopt, the value of the
   * `AWS_ACCESS_KEY_ID` environment variable is used.
   * @param aws_secret_access_key The AWS secret access key to use. If nullopt, the value of the
   * `AWS_SECRET_ACCESS_KEY` environment variable is used.
   * @param aws_endpoint_url Overwrite the endpoint url to use. If nullopt, the value of
   * the `AWS_ENDPOINT_URL` environment variable is used. If this is also not set, the regular
   * AWS url scheme is used: "https://<bucket>.s3.<region>.amazonaws.com/<object>"
   */
  S3Endpoint(std::string const& bucket_name,
             std::string const& object_name,
             std::optional<std::string> aws_region            = std::nullopt,
             std::optional<std::string> aws_access_key        = std::nullopt,
             std::optional<std::string> aws_secret_access_key = std::nullopt,
             std::optional<std::string> aws_endpoint_url      = std::nullopt)
    : S3Endpoint(url_from_bucket_and_object(
                   bucket_name, object_name, aws_region, std::move(aws_endpoint_url)),
                 std::move(aws_region),
                 std::move(aws_access_key),
                 std::move(aws_secret_access_key))
  {
  }

  void setopt(CurlHandle& curl) override
  {
    curl.setopt(CURLOPT_URL, _url.c_str());
    curl.setopt(CURLOPT_AWS_SIGV4, _aws_sigv4.c_str());
    curl.setopt(CURLOPT_USERPWD, _aws_userpwd.c_str());
  }
  std::string str() override { return _url; }
  ~S3Endpoint() override = default;
};

/**
 * @brief Handle of remote file.
 */
class RemoteHandle {
 private:
  std::unique_ptr<RemoteEndpoint> _endpoint;
  std::size_t _nbytes;

 public:
  /**
   * @brief Create a new remote handle from an endpoint and a file size.
   *
   * @param endpoint Remote endpoint used for subsequent IO.
   * @param nbytes The size of the remote file (in bytes).
   */
  RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint, std::size_t nbytes)
    : _endpoint{std::move(endpoint)}, _nbytes{nbytes}
  {
  }

  /**
   * @brief Create a new remote handle from an endpoint (infers the file size).
   *
   * The file size is received from the remote server using `endpoint`.
   *
   * @param endpoint Remote endpoint used for subsequently IO.
   */
  RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint)
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

  // A remote handle is moveable but not copyable.
  RemoteHandle(RemoteHandle&& o)               = default;
  RemoteHandle& operator=(RemoteHandle&& o)    = default;
  RemoteHandle(RemoteHandle const&)            = delete;
  RemoteHandle& operator=(RemoteHandle const&) = delete;

  /**
   * @brief Get the file size.
   *
   * Note, this is very fast, no communication needed.
   *
   * @return The number of bytes.
   */
  [[nodiscard]] std::size_t nbytes() const noexcept { return _nbytes; }

  /**
   * @brief Read from remote source into buffer (host or device memory).
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @return Number of bytes read, which is always `size`.
   */
  std::size_t read(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::read()", size);

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
      curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_host_memory);
    } else {
      curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_device_memory);
    }
    detail::CallbackContext ctx{buf, size};
    curl.setopt(CURLOPT_WRITEDATA, &ctx);

    try {
      if (is_host_mem) {
        curl.perform();
      } else {
        PushAndPopContext c(get_context_from_pointer(buf));
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

  /**
   * @brief Read from remote source into buffer (host or device memory) in parallel.
   *
   * This API is a parallel async version of `.read()` that partitions the operation
   * into tasks of size `task_size` for execution in the default thread pool.
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @param task_size Size of each task in bytes.
   * @return Future that on completion returns the size of bytes read, which is always `size`.
   */
  std::future<std::size_t> pread(void* buf,
                                 std::size_t size,
                                 std::size_t file_offset = 0,
                                 std::size_t task_size   = defaults::task_size())
  {
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::pread()", size);
    auto task = [this](void* devPtr_base,
                       std::size_t size,
                       std::size_t file_offset,
                       std::size_t devPtr_offset) -> std::size_t {
      return read(static_cast<char*>(devPtr_base) + devPtr_offset, size, file_offset);
    };
    return parallel_io(task, buf, size, file_offset, task_size, 0);
  }
};

}  // namespace kvikio
