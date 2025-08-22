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
#pragma once

#include <optional>
#include <string>

#include <curl/curl.h>

namespace kvikio::detail {
/**
 * @brief RAII wrapper for libcurl's URL handle (CURLU)
 *
 * This class provides automatic resource management for libcurl URL handles,
 * ensuring proper cleanup when the handle goes out of scope. The class is
 * move-only to prevent accidental sharing of the underlying resource.
 */
class CurlUrlHandle {
 private:
  CURLU* _handle{nullptr};

 public:
  /**
   * @brief Create a new libcurl URL handle
   *
   * @exception std::runtime_error if libcurl cannot allocate the handle (usually due to out of
   * memory)
   */
  CurlUrlHandle();

  /**
   * @brief Clean up the underlying URL handle
   */
  ~CurlUrlHandle() noexcept;

  CurlUrlHandle(CurlUrlHandle const&)            = delete;
  CurlUrlHandle& operator=(CurlUrlHandle const&) = delete;

  CurlUrlHandle(CurlUrlHandle&& other) noexcept;
  CurlUrlHandle& operator=(CurlUrlHandle&& other) noexcept;

  /**
   * @brief Get the underlying libcurl URL handle
   *
   * @return Pointer to the underlying libcurl URL handle
   * @note The returned pointer should not be freed manually as it is managed by this class
   */
  CURLU* get() const;
};

/**
 * @brief URL parsing utility using libcurl's URL API
 *
 * This class provides static methods for parsing URLs into their constituent
 * components (scheme, host, port, path, query, fragment).
 *
 * @note This class uses libcurl's URL parsing which follows RFC 3986
 *
 * Example usage:
 * @code
 * auto components = UrlParser::parse("https://example.com:8080/path?query=1#frag");
 * if (components.scheme.has_value()) {
 *     std::cout << "Scheme: " << components.scheme.value() << std::endl;
 * }
 * if (components.host.has_value()) {
 *     std::cout << "Host: " << components.host.value() << std::endl;
 * }
 * @endcode
 */
class UrlParser {
 public:
  /**
   * @brief Container for parsed URL components
   */
  struct UrlComponents {
    /**
     * @brief The URL scheme (e.g., "http", "https", "ftp"). May be empty for scheme-relative URLs
     * or paths.
     */
    std::optional<std::string> scheme;

    /**
     * @brief The hostname or IP address. May be empty for URLs without an authority component
     * (e.g., "file:///path").
     */
    std::optional<std::string> host;

    /**
     * @brief The port number as a string. Will be empty if no explicit port is specified in the
     * URL.
     * @note Default ports (e.g., 80 for HTTP, 443 for HTTPS) are not automatically filled in.
     */
    std::optional<std::string> port;

    /**
     * @brief The path component of the URL. Libcurl ensures that the path component is always
     * present, even if empty (will be "/" for URLs like "http://example.com").
     */
    std::optional<std::string> path;

    /**
     * @brief The query string (without the leading "?"). Empty if no query parameters are present.
     */
    std::optional<std::string> query;

    /**
     * @brief The fragment identifier (without the leading "#"). Empty if no fragment is present.
     */
    std::optional<std::string> fragment;
  };

  /**
   * @brief Parses the given URL according to RFC 3986 and extracts its components.
   *
   * @param url The URL string to parse
   * @param bitmask_url_flags Optional flags for URL parsing. Common flags include:
   *                          - CURLU_DEFAULT_SCHEME: Allows URLs without schemes
   *                          - CURLU_NON_SUPPORT_SCHEME: Accept non-supported schemes
   *                          - CURLU_URLENCODE: URL encode the path
   * @param bitmask_component_flags Optional flags for component extraction. Common flags include:
   *                                - CURLU_URLDECODE: URL decode the component
   *                                - CURLU_PUNYCODE: Return host as punycode
   *
   * @return UrlComponents structure containing the parsed URL components
   *
   * @throw std::runtime_error if the URL cannot be parsed or if component extraction fails
   *
   * Example:
   * @code
   * // Basic parsing
   * auto components = UrlParser::parse("https://api.example.com/v1/users?page=1");
   *
   * // Parsing with URL decoding
   * auto decoded = UrlParser::parse(
   *     "https://example.com/hello%20world",
   *     std::nullopt,
   *     CURLU_URLDECODE
   * );
   *
   * // Allow non-standard schemes
   * auto custom = UrlParser::parse(
   *     "myscheme://example.com",
   *     CURLU_NON_SUPPORT_SCHEME
   * );
   * @endcode
   */
  static UrlComponents parse(std::string const& url,
                             std::optional<unsigned int> bitmask_url_flags       = std::nullopt,
                             std::optional<unsigned int> bitmask_component_flags = std::nullopt);

  /**
   * @brief Extract a specific component from a parsed URL
   *
   * @param handle The CurlUrlHandle containing the parsed URL
   * @param part The URL part to extract (e.g., CURLUPART_SCHEME)
   * @param bitmask_component_flags Flags controlling extraction behavior
   * @param allowed_err_code Optional error code to treat as valid (e.g., CURLUE_NO_SCHEME)
   * @return The extracted component as a string, or std::nullopt if not present
   * @throw std::runtime_error if extraction fails with an unexpected error
   */
  static std::optional<std::string> extract_component(
    CurlUrlHandle const& handle,
    CURLUPart part,
    std::optional<unsigned int> bitmask_component_flags = std::nullopt,
    std::optional<CURLUcode> allowed_err_code           = std::nullopt);

  /**
   * @brief Extract a specific component from a URL string
   *
   * @param url The URL string from which to extract a component
   * @param part The URL part to extract
   * @param bitmask_url_flags Optional flags for URL parsing.
   * @param bitmask_component_flags Flags controlling extraction behavior
   * @param allowed_err_code Optional error code to treat as valid
   * @return The extracted component as a string, or std::nullopt if not present
   * @throw std::runtime_error if extraction fails with an unexpected error
   */
  static std::optional<std::string> extract_component(
    std::string const& url,
    CURLUPart part,
    std::optional<unsigned int> bitmask_url_flags       = std::nullopt,
    std::optional<unsigned int> bitmask_component_flags = std::nullopt,
    std::optional<CURLUcode> allowed_err_code           = std::nullopt);
};
}  // namespace kvikio::detail
