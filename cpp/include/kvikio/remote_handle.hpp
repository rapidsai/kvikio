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
#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
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
#include <kvikio/utils.hpp>

struct curl_slist;

namespace kvikio {

class CurlHandle;  // Prototype

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
  virtual std::string str() const = 0;

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
  HttpEndpoint(std::string url);
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
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
  curl_slist* _curl_header_list{};

  /**
   * @brief Unwrap an optional parameter, obtaining a default from the environment.
   *
   * If not nullopt, the optional's value is returned. Otherwise, the environment
   * variable `env_var` is used. If that also doesn't have a value:
   *   - if `err_msg` is empty, the empty string is returned.
   *   - if `err_msg` is not empty, `std::invalid_argument(`err_msg`)` is thrown.
   *
   * @param value The value to unwrap.
   * @param env_var The name of the environment variable to check if `value` isn't set.
   * @param err_msg The error message to throw on error or the empty string.
   * @return The parsed AWS argument or the empty string.
   */
  static std::string unwrap_or_default(std::optional<std::string> aws_arg,
                                       std::string const& env_var,
                                       std::string const& err_msg = "");

 public:
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
   * @param aws_endpoint_url Overwrite the endpoint url (including the protocol part) by using
   * the scheme: "<aws_endpoint_url>/<bucket_name>/<object_name>". If nullopt, the value of the
   * `AWS_ENDPOINT_URL` environment variable is used. If this is also not set, the regular AWS
   * url scheme is used: "https://<bucket_name>.s3.<region>.amazonaws.com/<object_name>".
   */
  static std::string url_from_bucket_and_object(std::string bucket_name,
                                                std::string object_name,
                                                std::optional<std::string> aws_region,
                                                std::optional<std::string> aws_endpoint_url);

  /**
   * @brief Given an url like "s3://<bucket>/<object>", return the name of the bucket and object.
   *
   * @throws std::invalid_argument if url is ill-formed or is missing the bucket or object name.
   *
   * @param s3_url S3 url.
   * @return Pair of strings: [bucket-name, object-name].
   */
  [[nodiscard]] static std::pair<std::string, std::string> parse_s3_url(std::string const& s3_url);

  /**
   * @brief Create a S3 endpoint from a url.
   *
   * @param url The full http url to the S3 file. NB: this should be an url starting with
   * "http://" or "https://". If you have an S3 url of the form "s3://<bucket>/<object>", please
   * use `S3Endpoint::parse_s3_url()` and `S3Endpoint::url_from_bucket_and_object() to convert it.
   * @param aws_region The AWS region, such as "us-east-1", to use. If nullopt, the value of the
   * `AWS_DEFAULT_REGION` environment variable is used.
   * @param aws_access_key The AWS access key to use. If nullopt, the value of the
   * `AWS_ACCESS_KEY_ID` environment variable is used.
   * @param aws_secret_access_key The AWS secret access key to use. If nullopt, the value of the
   * `AWS_SECRET_ACCESS_KEY` environment variable is used.
   * @param aws_session_token The AWS session token to use. If nullopt, the value of the
   * `AWS_SESSION_TOKEN` environment variable is used.
   */
  S3Endpoint(std::string url,
             std::optional<std::string> aws_region            = std::nullopt,
             std::optional<std::string> aws_access_key        = std::nullopt,
             std::optional<std::string> aws_secret_access_key = std::nullopt,
             std::optional<std::string> aws_session_token     = std::nullopt);

  /**
   * @brief Create a S3 endpoint from a bucket and object name.
   *
   * @param bucket_and_object_names The bucket and object names of the S3 bucket.
   * @param aws_region The AWS region, such as "us-east-1", to use. If nullopt, the value of the
   * `AWS_DEFAULT_REGION` environment variable is used.
   * @param aws_access_key The AWS access key to use. If nullopt, the value of the
   * `AWS_ACCESS_KEY_ID` environment variable is used.
   * @param aws_secret_access_key The AWS secret access key to use. If nullopt, the value of the
   * `AWS_SECRET_ACCESS_KEY` environment variable is used.
   * @param aws_endpoint_url Overwrite the endpoint url (including the protocol part) by using
   * the scheme: "<aws_endpoint_url>/<bucket_name>/<object_name>". If nullopt, the value of the
   * `AWS_ENDPOINT_URL` environment variable is used. If this is also not set, the regular AWS
   * url scheme is used: "https://<bucket_name>.s3.<region>.amazonaws.com/<object_name>".
   * @param aws_session_token The AWS session token to use. If nullopt, the value of the
   * `AWS_SESSION_TOKEN` environment variable is used.
   */
  S3Endpoint(std::pair<std::string, std::string> bucket_and_object_names,
             std::optional<std::string> aws_region            = std::nullopt,
             std::optional<std::string> aws_access_key        = std::nullopt,
             std::optional<std::string> aws_secret_access_key = std::nullopt,
             std::optional<std::string> aws_endpoint_url      = std::nullopt,
             std::optional<std::string> aws_session_token     = std::nullopt);

  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  ~S3Endpoint() override;
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
  RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint, std::size_t nbytes);

  /**
   * @brief Create a new remote handle from an endpoint (infers the file size).
   *
   * The file size is received from the remote server using `endpoint`.
   *
   * @param endpoint Remote endpoint used for subsequently IO.
   */
  RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint);

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
  [[nodiscard]] std::size_t nbytes() const noexcept;

  /**
   * @brief Get a const reference to the underlying remote endpoint.
   *
   * @return The remote endpoint.
   */
  [[nodiscard]] RemoteEndpoint const& endpoint() const noexcept;

  /**
   * @brief Read from remote source into buffer (host or device memory).
   *
   * When reading into device memory, a bounce buffer is used to avoid many small memory
   * copies to device. Use `kvikio::default::bounce_buffer_size_reset()` to set the size
   * of this bounce buffer (default 16 MiB).
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @return Number of bytes read, which is always `size`.
   */
  std::size_t read(void* buf, std::size_t size, std::size_t file_offset = 0);

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
                                 std::size_t task_size   = defaults::task_size());
};

}  // namespace kvikio
