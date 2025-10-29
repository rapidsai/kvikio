/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/utils.hpp>

struct curl_slist;

namespace kvikio {

class CurlHandle;  // Prototype

/**
 * @brief Types of remote file endpoints supported by KvikIO.
 *
 * This enum defines the different protocols and services that can be used to access remote files.
 * It is used to specify or detect the type of remote endpoint when opening files.
 */
enum class RemoteEndpointType : uint8_t {
  AUTO,  ///< Automatically detect the endpoint type from the URL. KvikIO will attempt to infer the
         ///< appropriate protocol based on the URL format.
  S3,    ///< AWS S3 endpoint using credentials-based authentication. Requires AWS environment
         ///< variables (such as AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) to be
         ///< set.
  S3_PUBLIC,  ///< AWS S3 endpoint for publicly accessible objects. No credentials required as the
              ///< objects have public read permissions enabled. Used for open datasets and public
              ///< buckets.
  S3_PRESIGNED_URL,  ///< AWS S3 endpoint using a presigned URL. No credentials required as
                     ///< authentication is embedded in the URL with time-limited access.
  WEBHDFS,  ///< Apache Hadoop WebHDFS (Web-based Hadoop Distributed File System) endpoint for
            ///< accessing files stored in HDFS over HTTP/HTTPS.
  HTTP,  ///< Generic HTTP or HTTPS endpoint for accessing files from web servers. This is used for
         ///< standard web resources that do not fit the other specific categories.
};

/**
 * @brief Abstract base class for remote endpoints.
 *
 * In this context, an endpoint refers to a remote file using a specific communication protocol.
 *
 * Each communication protocol, such as HTTP or S3, needs to implement this ABC and implement
 * its own ctor that takes communication protocol specific arguments.
 */
class RemoteEndpoint {
 protected:
  RemoteEndpointType _remote_endpoint_type{RemoteEndpointType::AUTO};
  RemoteEndpoint(RemoteEndpointType remote_endpoint_type);

 public:
  virtual ~RemoteEndpoint() = default;

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

  /**
   * @brief Get the size of the remote file.
   *
   * @return The file size
   */
  virtual std::size_t get_file_size() = 0;

  /**
   * @brief Set up the range request in order to read part of a file given the file offset and read
   * size.
   */
  virtual void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) = 0;

  /**
   * @brief Get the type of the remote file.
   *
   * @return The type of the remote file.
   */
  [[nodiscard]] RemoteEndpointType remote_endpoint_type() const noexcept;
};

/**
 * @brief A remote endpoint for HTTP/HTTPS resources
 *
 * This endpoint is for accessing files via standard HTTP/HTTPS protocols without any specialized
 * authentication.
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

  ~HttpEndpoint() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for HTTP/HTTPS endpoints.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};

/**
 * @brief A remote endpoint for AWS S3 storage requiring credentials
 *
 * This endpoint is for accessing private S3 objects using AWS credentials (access key, secret key,
 * region and optional session token).
 */
class S3Endpoint : public RemoteEndpoint {
 private:
  std::string _url;
  std::string _aws_sigv4;
  std::string _aws_userpwd;
  curl_slist* _curl_header_list{};

 public:
  /**
   * @brief Get url from a AWS S3 bucket and object name.
   *
   * @exception std::invalid_argument if no region is specified and no default region is
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
   * @exception std::invalid_argument if url is ill-formed or is missing the bucket or object name.
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

  ~S3Endpoint() override;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for S3 endpoints (excluding presigned URL).
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};

/**
 * @brief A remote endpoint for publicly accessible S3 objects without authentication
 *
 * This endpoint is for accessing S3 objects configured with public read permissions,
 * requiring no authentication. Supports AWS S3 services with anonymous access enabled.
 */
class S3PublicEndpoint : public RemoteEndpoint {
 private:
  std::string _url;

 public:
  explicit S3PublicEndpoint(std::string url);

  ~S3PublicEndpoint() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for S3 public endpoints.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};

/**
 * @brief A remote endpoint for AWS S3 storage using presigned URLs.
 *
 * This endpoint is for accessing S3 objects via presigned URLs, which provide time-limited access
 * without requiring AWS credentials on the client side.
 */
class S3EndpointWithPresignedUrl : public RemoteEndpoint {
 private:
  std::string _url;

 public:
  explicit S3EndpointWithPresignedUrl(std::string presigned_url);

  ~S3EndpointWithPresignedUrl() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for S3 endpoints with presigned URL.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
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
   * @brief Create a remote file handle from a URL.
   *
   * This function creates a RemoteHandle for reading data from various remote endpoints
   * including HTTP/HTTPS servers, AWS S3 buckets, S3 presigned URLs, and WebHDFS.
   * The endpoint type can be automatically detected from the URL or explicitly specified.
   *
   * @param url The URL of the remote file. Supported formats include:
   *   - S3 with credentials
   *   - S3 presigned URL
   *   - WebHDFS
   *   - HTTP/HTTPS
   * @param remote_endpoint_type The type of remote endpoint. Default is RemoteEndpointType::AUTO
   * which automatically detects the endpoint type from the URL. Can be explicitly set to
   * RemoteEndpointType::S3, RemoteEndpointType::S3_PRESIGNED_URL, RemoteEndpointType::WEBHDFS, or
   * RemoteEndpointType::HTTP to force a specific endpoint type.
   * @param allow_list Optional list of allowed endpoint types. If provided:
   *   - If remote_endpoint_type is RemoteEndpointType::AUTO, Types are tried in the exact order
   *     specified until a match is found.
   *   - In explicit mode, the specified type must be in this list, otherwise an exception is
   *     thrown.
   *
   * If not provided, defaults to all supported types in this order: RemoteEndpointType::S3,
   * RemoteEndpointType::S3_PRESIGNED_URL, RemoteEndpointType::WEBHDFS, and
   * RemoteEndpointType::HTTP.
   * @param nbytes Optional file size in bytes. If not provided, the function sends additional
   * request to the server to query the file size.
   * @return A RemoteHandle object that can be used to read data from the remote file.
   * @exception std::runtime_error If:
   *   - If the URL is malformed or missing required components.
   *   - RemoteEndpointType::AUTO mode is used and the URL doesn't match any supported endpoint
   * type.
   *   - The specified endpoint type is not in the `allow_list`.
   *   - The URL is invalid for the specified endpoint type.
   *   - Unable to connect to the remote server or determine file size (when nbytes not provided).
   *
   * Example:
   * - Auto-detect endpoint type from URL
   *   @code{.cpp}
   *   auto handle = kvikio::RemoteHandle::open(
   *       "https://bucket.s3.amazonaws.com/object?X-Amz-Algorithm=AWS4-HMAC-SHA256"
   *       "&X-Amz-Credential=...&X-Amz-Signature=..."
   *   );
   *   @endcode
   *
   * - Open S3 file with explicit endpoint type
   *   @code{.cpp}
   *
   *   auto handle = kvikio::RemoteHandle::open(
   *       "https://my-bucket.s3.us-east-1.amazonaws.com/data.bin",
   *       kvikio::RemoteEndpointType::S3
   *   );
   *   @endcode
   *
   * - Restrict endpoint type candidates
   *   @code{.cpp}
   *   std::vector<kvikio::RemoteEndpointType> allow_list = {
   *       kvikio::RemoteEndpointType::HTTP,
   *       kvikio::RemoteEndpointType::S3_PRESIGNED_URL
   *   };
   *   auto handle = kvikio::RemoteHandle::open(
   *       user_provided_url,
   *       kvikio::RemoteEndpointType::AUTO,
   *       allow_list
   *   );
   *   @endcode
   *
   * - Provide known file size to skip HEAD request
   *   @code{.cpp}
   *   auto handle = kvikio::RemoteHandle::open(
   *       "https://example.com/large-file.bin",
   *       kvikio::RemoteEndpointType::HTTP,
   *       std::nullopt,
   *       1024 * 1024 * 100  // 100 MB
   *   );
   *   @endcode
   */
  static RemoteHandle open(std::string url,
                           RemoteEndpointType remote_endpoint_type = RemoteEndpointType::AUTO,
                           std::optional<std::vector<RemoteEndpointType>> allow_list = std::nullopt,
                           std::optional<std::size_t> nbytes = std::nullopt);

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
   * @brief Get the type of the remote file.
   *
   * @return The type of the remote file.
   */
  [[nodiscard]] RemoteEndpointType remote_endpoint_type() const noexcept;

  /**
   * @brief Get the file size.
   *
   * Note, the file size is retrieved at construction so this method is very fast, no communication
   * needed.
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
