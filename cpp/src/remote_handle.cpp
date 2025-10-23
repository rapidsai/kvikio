/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/parallel_operation.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include <kvikio/hdfs.hpp>
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
    KVIKIO_NVTX_FUNC_RANGE();
  }

  /**
   * @brief The bounce buffer if flushed to device on destruction.
   */
  ~BounceBufferH2D() noexcept
  {
    KVIKIO_NVTX_FUNC_RANGE();
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
    KVIKIO_NVTX_FUNC_RANGE();
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
    KVIKIO_NVTX_FUNC_RANGE();
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
    KVIKIO_NVTX_FUNC_RANGE();
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

/**
 * @brief Get the file size, if using `HEAD` request to obtain the content-length header is
 * permitted.
 *
 * This function works for the `HttpEndpoint` and `S3Endpoint`, but not for
 * `S3EndpointWithPresignedUrl`, which does not allow `HEAD` request.
 *
 * @param endpoint The remote endpoint
 * @param url The URL of the remote file
 * @return The file size
 */
std::size_t get_file_size_using_head_impl(RemoteEndpoint& endpoint, std::string const& url)
{
  auto curl = create_curl_handle();

  endpoint.setopt(curl);
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
  curl.perform();
  curl_off_t cl;
  curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
  KVIKIO_EXPECT(
    cl >= 0,
    "cannot get size of " + endpoint.str() + ", content-length not provided by the server",
    std::runtime_error);
  return static_cast<std::size_t>(cl);
}

/**
 * @brief Set up the range request for libcurl. Use this method when HTTP range request is supposed.
 *
 * @param curl A curl handle
 * @param file_offset File offset
 * @param size read size
 */
void setup_range_request_impl(CurlHandle& curl, std::size_t file_offset, std::size_t size)
{
  std::string const byte_range =
    std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
  curl.setopt(CURLOPT_RANGE, byte_range.c_str());
}

/**
 * @brief Whether the given URL is compatible with the S3 endpoint (including the credential-based
 * access and presigned URL) which uses HTTP/HTTPS.
 *
 * @param url A URL.
 * @return Boolean answer.
 */
bool url_has_aws_s3_http_format(std::string const& url)
{
  // Currently KvikIO supports the following AWS S3 HTTP URL formats:
  static std::array const s3_patterns = {
    // Virtual host style: https://<bucket-name>.s3.<region-code>.amazonaws.com/<object-key-name>
    std::regex(R"(https?://[^/]+\.s3\.[^.]+\.amazonaws\.com/.+$)", std::regex_constants::icase),

    // Path style (deprecated but still popular):
    // https://s3.<region-code>.amazonaws.com/<bucket-name>/<object-key-name>
    std::regex(R"(https?://s3\.[^.]+\.amazonaws\.com/[^/]+/.+$)", std::regex_constants::icase),

    // Legacy global endpoint: no region code
    std::regex(R"(https?://[^/]+\.s3\.amazonaws\.com/.+$)", std::regex_constants::icase),
    std::regex(R"(https?://s3\.amazonaws\.com/[^/]+/.+$)", std::regex_constants::icase),

    // Legacy regional endpoint: s3 and region code are delimited by - instead of .
    std::regex(R"(https?://[^/]+\.s3-[^.]+\.amazonaws\.com/.+$)", std::regex_constants::icase),
    std::regex(R"(https?://s3-[^.]+\.amazonaws\.com/[^/]+/.+$)", std::regex_constants::icase)};

  return std::any_of(s3_patterns.begin(), s3_patterns.end(), [&url = url](auto const& pattern) {
    std::smatch match_result;
    return std::regex_match(url, match_result, pattern);
  });
}

char const* get_remote_endpoint_type_name(RemoteEndpointType remote_endpoint_type)
{
  switch (remote_endpoint_type) {
    case RemoteEndpointType::S3: return "S3";
    case RemoteEndpointType::S3_PUBLIC: return "S3 public";
    case RemoteEndpointType::S3_PRESIGNED_URL: return "S3 with presigned URL";
    case RemoteEndpointType::WEBHDFS: return "WebHDFS";
    case RemoteEndpointType::HTTP: return "HTTP";
    case RemoteEndpointType::AUTO: return "AUTO";
    default:
      // Unreachable
      KVIKIO_FAIL("Unknown RemoteEndpointType: " +
                  std::to_string(static_cast<int>(remote_endpoint_type)));
      return "UNKNOWN";
  }
}

std::string encode_special_chars_in_path(std::string const& url)
{
  auto components = detail::UrlParser::parse(url);
  components.path = detail::UrlEncoder::encode_path(components.path.value());
  return detail::UrlBuilder::build_manually(components);
}
}  // namespace

RemoteEndpoint::RemoteEndpoint(RemoteEndpointType remote_endpoint_type)
  : _remote_endpoint_type{remote_endpoint_type}
{
}

RemoteEndpointType RemoteEndpoint::remote_endpoint_type() const noexcept
{
  return _remote_endpoint_type;
}

HttpEndpoint::HttpEndpoint(std::string url)
  : RemoteEndpoint{RemoteEndpointType::HTTP}, _url{std::move(url)}
{
}

std::string HttpEndpoint::str() const { return _url; }

std::size_t HttpEndpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  return get_file_size_using_head_impl(*this, _url);
}

void HttpEndpoint::setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size)
{
  setup_range_request_impl(curl, file_offset, size);
}

bool HttpEndpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    auto parsed_url = detail::UrlParser::parse(url);
    if ((parsed_url.scheme != "http") && (parsed_url.scheme != "https")) { return false; };

    // Check whether the file path exists, excluding the leading "/"
    return parsed_url.path->length() > 1;
  } catch (...) {
    return false;
  }
}

void HttpEndpoint::setopt(CurlHandle& curl) { curl.setopt(CURLOPT_URL, _url.c_str()); }

void S3Endpoint::setopt(CurlHandle& curl)
{
  auto new_url = encode_special_chars_in_path(_url);
  curl.setopt(CURLOPT_URL, new_url.c_str());

  curl.setopt(CURLOPT_AWS_SIGV4, _aws_sigv4.c_str());
  curl.setopt(CURLOPT_USERPWD, _aws_userpwd.c_str());
  if (_curl_header_list) { curl.setopt(CURLOPT_HTTPHEADER, _curl_header_list); }
}

std::string S3Endpoint::unwrap_or_default(std::optional<std::string> aws_arg,
                                          std::string const& env_var,
                                          std::string const& err_msg)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (aws_arg.has_value()) { return std::move(*aws_arg); }

  char const* env = std::getenv(env_var.c_str());
  if (env == nullptr) {
    if (err_msg.empty()) { return std::string(); }
    KVIKIO_FAIL(err_msg, std::invalid_argument);
  }
  return std::string(env);
}

std::string S3Endpoint::url_from_bucket_and_object(std::string bucket_name,
                                                   std::string object_name,
                                                   std::optional<std::string> aws_region,
                                                   std::optional<std::string> aws_endpoint_url)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const endpoint_url = unwrap_or_default(std::move(aws_endpoint_url), "AWS_ENDPOINT_URL");
  std::stringstream ss;
  if (endpoint_url.empty()) {
    auto const region =
      unwrap_or_default(std::move(aws_region),
                        "AWS_DEFAULT_REGION",
                        "S3: must provide `aws_region` if AWS_DEFAULT_REGION isn't set.");
    // "s3" is a non-standard URI scheme used by AWS CLI and AWS SDK, and cannot be identified by
    // libcurl. A valid HTTP/HTTPS URL needs to be constructed for use in libcurl. Here the AWS
    // virtual host style is used.
    ss << "https://" << bucket_name << ".s3." << region << ".amazonaws.com/" << object_name;
  } else {
    ss << endpoint_url << "/" << bucket_name << "/" << object_name;
  }
  return ss.str();
}

std::pair<std::string, std::string> S3Endpoint::parse_s3_url(std::string const& s3_url)
{
  KVIKIO_NVTX_FUNC_RANGE();
  // Regular expression to match s3://<bucket>/<object>
  std::regex static const pattern{R"(^s3://([^/]+)/(.+))", std::regex_constants::icase};
  std::smatch matches;
  if (std::regex_match(s3_url, matches, pattern)) { return {matches[1].str(), matches[2].str()}; }
  KVIKIO_FAIL("Input string does not match the expected S3 URL format.", std::invalid_argument);
  return {};
}

S3Endpoint::S3Endpoint(std::string url,
                       std::optional<std::string> aws_region,
                       std::optional<std::string> aws_access_key,
                       std::optional<std::string> aws_secret_access_key,
                       std::optional<std::string> aws_session_token)
  : RemoteEndpoint{RemoteEndpointType::S3}, _url{std::move(url)}
{
  KVIKIO_NVTX_FUNC_RANGE();
  // Regular expression to match http[s]://
  std::regex static const pattern{R"(^https?://.*)", std::regex_constants::icase};
  KVIKIO_EXPECT(std::regex_search(_url, pattern),
                "url must start with http:// or https://",
                std::invalid_argument);

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
  // Access key IDs beginning with ASIA are temporary credentials that are created using AWS STS
  // operations. They need a session token to work.
  if (access_key.compare(0, 4, std::string("ASIA")) == 0) {
    // Create a Custom Curl header for the session token.
    // The _curl_header_list created by curl_slist_append must be manually freed
    // (see https://curl.se/libcurl/c/CURLOPT_HTTPHEADER.html)
    auto session_token =
      unwrap_or_default(std::move(aws_session_token),
                        "AWS_SESSION_TOKEN",
                        "When using temporary credentials, AWS_SESSION_TOKEN must be set.");
    std::stringstream ss;
    ss << "x-amz-security-token: " << session_token;
    _curl_header_list = curl_slist_append(NULL, ss.str().c_str());
    KVIKIO_EXPECT(_curl_header_list != nullptr,
                  "Failed to create curl header for AWS token",
                  std::runtime_error);
  }
}

S3Endpoint::S3Endpoint(std::pair<std::string, std::string> bucket_and_object_names,
                       std::optional<std::string> aws_region,
                       std::optional<std::string> aws_access_key,
                       std::optional<std::string> aws_secret_access_key,
                       std::optional<std::string> aws_endpoint_url,
                       std::optional<std::string> aws_session_token)
  : S3Endpoint(url_from_bucket_and_object(std::move(bucket_and_object_names.first),
                                          std::move(bucket_and_object_names.second),
                                          aws_region,
                                          std::move(aws_endpoint_url)),
               aws_region,
               std::move(aws_access_key),
               std::move(aws_secret_access_key),
               std::move(aws_session_token))
{
  KVIKIO_NVTX_FUNC_RANGE();
}

S3Endpoint::~S3Endpoint() { curl_slist_free_all(_curl_header_list); }

std::string S3Endpoint::str() const { return _url; }

std::size_t S3Endpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  return get_file_size_using_head_impl(*this, _url);
}

void S3Endpoint::setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  setup_range_request_impl(curl, file_offset, size);
}

bool S3Endpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    auto parsed_url = detail::UrlParser::parse(url, CURLU_NON_SUPPORT_SCHEME);

    if (parsed_url.scheme == "s3") {
      if (!parsed_url.host.has_value()) { return false; }
      if (!parsed_url.path.has_value()) { return false; }

      // Check whether the S3 object key exists
      std::regex static const pattern(R"(^/.+$)");
      return std::regex_search(parsed_url.path.value(), pattern);
    } else if ((parsed_url.scheme == "http") || (parsed_url.scheme == "https")) {
      return url_has_aws_s3_http_format(url) && !S3EndpointWithPresignedUrl::is_url_valid(url);
    }
  } catch (...) {
  }
  return false;
}

S3PublicEndpoint::S3PublicEndpoint(std::string url)
  : RemoteEndpoint{RemoteEndpointType::S3_PUBLIC}, _url{std::move(url)}
{
}

void S3PublicEndpoint::setopt(CurlHandle& curl)
{
  auto new_url = encode_special_chars_in_path(_url);
  curl.setopt(CURLOPT_URL, new_url.c_str());
}

std::string S3PublicEndpoint::str() const { return _url; }

std::size_t S3PublicEndpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  return get_file_size_using_head_impl(*this, _url);
}

void S3PublicEndpoint::setup_range_request(CurlHandle& curl,
                                           std::size_t file_offset,
                                           std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  setup_range_request_impl(curl, file_offset, size);
}

bool S3PublicEndpoint::is_url_valid(std::string const& url) noexcept
{
  return S3Endpoint::is_url_valid(url);
}

S3EndpointWithPresignedUrl::S3EndpointWithPresignedUrl(std::string presigned_url)
  : RemoteEndpoint{RemoteEndpointType::S3_PRESIGNED_URL}, _url{std::move(presigned_url)}
{
}

void S3EndpointWithPresignedUrl::setopt(CurlHandle& curl)
{
  curl.setopt(CURLOPT_URL, _url.c_str());
}

std::string S3EndpointWithPresignedUrl::str() const { return _url; }

namespace {
/**
 * @brief Callback for the `CURLOPT_HEADERFUNCTION` parameter in libcurl
 *
 * The header callback is called once for each header and only complete header lines are passed on
 * to the callback. The provided header line is not null-terminated.
 *
 * @param data Transfer buffer where new data is received
 * @param size Curl internal implementation always sets this parameter to 1
 * @param num_bytes The size of new data received
 * @param userdata User-defined data
 * @return The number of bytes consumed by the callback
 * @exception std::invalid_argument if the server does not know the file size, thereby using "*" as
 * the filler text in the content-range header of the HTTP message.
 */
std::size_t callback_header(char* data, std::size_t size, std::size_t num_bytes, void* userdata)
{
  auto new_data_size = size * num_bytes;
  auto* file_size    = reinterpret_cast<long*>(userdata);

  // The header line is not null-terminated. This constructor overload ensures header_line.data() is
  // null-terminated.
  std::string const header_line{data, new_data_size};

  // The content-range header has the format
  // Content-Range: <unit> <range>/<size>
  // Content-Range: <unit> <range>/*
  // Content-Range: <unit> */<size>
  std::regex static const pattern(R"(Content-Range:[^/]+/(.*))", std::regex::icase);
  std::smatch match_result;
  bool found = std::regex_search(header_line, match_result, pattern);
  if (found) {
    // If the file size is unknown (represented by "*" in the content-range header), string-to-long
    // conversion will throw an `std::invalid_argument` exception. The exception message from
    // `std::stol` is usually too concise to be useful (being simply a string of "stol"), so a
    // custom exception is used instead.
    try {
      *file_size = std::stol(match_result[1].str());
    } catch (...) {
      KVIKIO_FAIL("File size information missing on the server side.", std::invalid_argument);
    }
  }
  return new_data_size;
}
}  // namespace

std::size_t S3EndpointWithPresignedUrl::get_file_size()
{
  // Usually the `HEAD` request is used to obtain the content-length (file size). However, AWS S3
  // does not allow it for presigned URL. The workaround here is to send the `GET` request with
  // 1-byte range, so that we can still obtain the header information at a negligible cost. Since
  // the content-length header is now at a fixed value of 1, we instead extract the file size value
  // from content-range.

  KVIKIO_NVTX_FUNC_RANGE();

  auto curl = create_curl_handle();
  curl.setopt(CURLOPT_URL, _url.c_str());

  // 1-byte range, specified in the format "<start-byte>-<end-byte>""
  std::string my_range{"0-0"};
  curl.setopt(CURLOPT_RANGE, my_range.c_str());

  long file_size{};
  curl.setopt(CURLOPT_HEADERDATA, static_cast<void*>(&file_size));
  curl.setopt(CURLOPT_HEADERFUNCTION, callback_header);

  curl.perform();
  return file_size;
}

void S3EndpointWithPresignedUrl::setup_range_request(CurlHandle& curl,
                                                     std::size_t file_offset,
                                                     std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  setup_range_request_impl(curl, file_offset, size);
}

bool S3EndpointWithPresignedUrl::is_url_valid(std::string const& url) noexcept
{
  try {
    if (!url_has_aws_s3_http_format(url)) { return false; }

    auto parsed_url = detail::UrlParser::parse(url);
    if (!parsed_url.query.has_value()) { return false; }

    // Reference: https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html
    return parsed_url.query->find("X-Amz-Algorithm") != std::string::npos &&
           parsed_url.query->find("X-Amz-Signature") != std::string::npos;
  } catch (...) {
    return false;
  }
}

RemoteHandle RemoteHandle::open(std::string url,
                                RemoteEndpointType remote_endpoint_type,
                                std::optional<std::vector<RemoteEndpointType>> allow_list,
                                std::optional<std::size_t> nbytes)
{
  if (!allow_list.has_value()) {
    allow_list = {RemoteEndpointType::S3,
                  RemoteEndpointType::S3_PUBLIC,
                  RemoteEndpointType::S3_PRESIGNED_URL,
                  RemoteEndpointType::WEBHDFS,
                  RemoteEndpointType::HTTP};
  }

  auto const scheme =
    detail::UrlParser::extract_component(url, CURLUPART_SCHEME, CURLU_NON_SUPPORT_SCHEME);
  KVIKIO_EXPECT(scheme.has_value(), "Missing scheme in URL.");

  // Helper to create endpoint based on type
  auto create_endpoint =
    [&url = url, &scheme = scheme](RemoteEndpointType type) -> std::unique_ptr<RemoteEndpoint> {
    switch (type) {
      case RemoteEndpointType::S3:
        if (!S3Endpoint::is_url_valid(url)) { return nullptr; }
        if (scheme.value() == "s3") {
          auto const [bucket, object] = S3Endpoint::parse_s3_url(url);
          return std::make_unique<S3Endpoint>(std::pair{bucket, object});
        }
        return std::make_unique<S3Endpoint>(url);

      case RemoteEndpointType::S3_PUBLIC:
        if (!S3PublicEndpoint::is_url_valid(url)) { return nullptr; }
        return std::make_unique<S3PublicEndpoint>(url);

      case RemoteEndpointType::S3_PRESIGNED_URL:
        if (!S3EndpointWithPresignedUrl::is_url_valid(url)) { return nullptr; }
        return std::make_unique<S3EndpointWithPresignedUrl>(url);

      case RemoteEndpointType::WEBHDFS:
        if (!WebHdfsEndpoint::is_url_valid(url)) { return nullptr; }
        return std::make_unique<WebHdfsEndpoint>(url);

      case RemoteEndpointType::HTTP:
        if (!HttpEndpoint::is_url_valid(url)) { return nullptr; }
        return std::make_unique<HttpEndpoint>(url);

      default: return nullptr;
    }
  };

  std::unique_ptr<RemoteEndpoint> endpoint;

  if (remote_endpoint_type == RemoteEndpointType::AUTO) {
    // Try each allowed type in the order of allowlist
    for (auto const& type : allow_list.value()) {
      try {
        endpoint = create_endpoint(type);
        if (endpoint == nullptr) { continue; }
        if (type == RemoteEndpointType::S3) {
          // Check connectivity for the credential-based S3 endpoint, and throw an exception if
          // failed
          endpoint->get_file_size();
        }
      } catch (...) {
        // If the credential-based S3 endpoint cannot be used to access the URL, try using S3 public
        // endpoint instead if it is in the allowlist
        if (type == RemoteEndpointType::S3 &&
            std::find(allow_list->begin(), allow_list->end(), RemoteEndpointType::S3_PUBLIC) !=
              allow_list->end()) {
          endpoint = std::make_unique<S3PublicEndpoint>(url);
        } else {
          throw;
        }
      }

      // At this point, a matching endpoint has been found
      break;
    }
    KVIKIO_EXPECT(endpoint.get() != nullptr, "Unsupported endpoint URL.", std::runtime_error);
  } else {
    // Validate it is in the allow list
    KVIKIO_EXPECT(
      std::find(allow_list->begin(), allow_list->end(), remote_endpoint_type) != allow_list->end(),
      std::string{get_remote_endpoint_type_name(remote_endpoint_type)} +
        " is not in the allowlist.",
      std::runtime_error);

    // Create the specific type
    endpoint = create_endpoint(remote_endpoint_type);
    KVIKIO_EXPECT(endpoint.get() != nullptr,
                  std::string{"Invalid URL for "} +
                    get_remote_endpoint_type_name(remote_endpoint_type) + " endpoint",
                  std::runtime_error);
  }

  return nbytes.has_value() ? RemoteHandle(std::move(endpoint), nbytes.value())
                            : RemoteHandle(std::move(endpoint));
}

RemoteHandle::RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint, std::size_t nbytes)
  : _endpoint{std::move(endpoint)}, _nbytes{nbytes}
{
  KVIKIO_NVTX_FUNC_RANGE();
}

RemoteHandle::RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint)
{
  KVIKIO_NVTX_FUNC_RANGE();
  _nbytes   = endpoint->get_file_size();
  _endpoint = std::move(endpoint);
}

RemoteEndpointType RemoteHandle::remote_endpoint_type() const noexcept
{
  return _endpoint->remote_endpoint_type();
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
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);
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
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);

  ctx->bounce_buffer->write(data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}
}  // namespace

std::size_t RemoteHandle::read(void* buf, std::size_t size, std::size_t file_offset)
{
  KVIKIO_NVTX_FUNC_RANGE(size);

  if (file_offset + size > _nbytes) {
    std::stringstream ss;
    ss << "cannot read " << file_offset << "+" << size << " bytes into a " << _nbytes
       << " bytes file (" << _endpoint->str() << ")";
    KVIKIO_FAIL(ss.str(), std::invalid_argument);
  }
  bool const is_host_mem = is_host_memory(buf);
  auto curl              = create_curl_handle();
  _endpoint->setopt(curl);
  _endpoint->setup_range_request(curl, file_offset, size);

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
      KVIKIO_FAIL(ss.str(), std::overflow_error);
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
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size);
  auto task = [this](void* devPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t devPtr_offset) -> std::size_t {
    return read(static_cast<char*>(devPtr_base) + devPtr_offset, size, file_offset);
  };
  return parallel_io(task, buf, size, file_offset, task_size, 0, call_idx, nvtx_color);
}

}  // namespace kvikio
