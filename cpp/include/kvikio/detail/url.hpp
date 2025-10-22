/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 * @note This class uses libcurl's URL parsing which follows RFC 3986 plus. See
 * https://curl.se/docs/url-syntax.html
 *
 * Example:
 * @code{.cpp}
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
   * @brief Parses the given URL according to RFC 3986 plus and extracts its components.
   *
   * @param url The URL string to parse
   * @param bitmask_url_flags Optional flags for URL parsing. Common flags include:
   *  - CURLU_DEFAULT_SCHEME: Allows URLs without schemes
   *  - CURLU_NON_SUPPORT_SCHEME: Accept non-supported schemes
   *  - CURLU_URLENCODE: URL encode the path
   * @param bitmask_component_flags Optional flags for component extraction. Common flags include:
   *  - CURLU_URLDECODE: URL decode the component
   *  - CURLU_PUNYCODE: Return host as punycode
   *
   * @return UrlComponents structure containing the parsed URL components
   *
   * @exception std::runtime_error if the URL cannot be parsed or if component extraction fails
   *
   * Example:
   * @code{.cpp}
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
   * // Allow non-standard schemes, i.e. schemes not registered with Internet Assigned Numbers
   * // Authority (IANA), such as AWS S3
   * auto custom = UrlParser::parse(
   *     "s3://my-bucket/my-object.bin",
   *     CURLU_NON_SUPPORT_SCHEME
   * );
   * @endcode
   */
  static UrlComponents parse(std::string const& url,
                             std::optional<unsigned int> bitmask_url_flags       = std::nullopt,
                             std::optional<unsigned int> bitmask_component_flags = std::nullopt);

  /**
   * @brief Extract a specific component from a CurlUrlHandle
   *
   * @param handle The CurlUrlHandle containing the parsed URL
   * @param part The URL part to extract (e.g., CURLUPART_SCHEME)
   * @param bitmask_component_flags Flags controlling extraction behavior
   * @param allowed_err_code Optional error code to treat as valid (e.g., CURLUE_NO_SCHEME)
   * @return The extracted component as a string, or std::nullopt if not present
   * @exception std::runtime_error if extraction fails with an unexpected error
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
   * @exception std::runtime_error if extraction fails with an unexpected error
   */
  static std::optional<std::string> extract_component(
    std::string const& url,
    CURLUPart part,
    std::optional<unsigned int> bitmask_url_flags       = std::nullopt,
    std::optional<unsigned int> bitmask_component_flags = std::nullopt,
    std::optional<CURLUcode> allowed_err_code           = std::nullopt);
};

/**
 * @brief URL builder utility using libcurl's URL API
 *
 * This class provides methods for constructing and modifying URLs by setting individual components
 * (scheme, host, port, path, query, fragment).
 *
 * @note This class uses libcurl's URL parsing which follows RFC 3986 plus. See
 * https://curl.se/docs/url-syntax.html
 *
 * Example:
 * @code{.cpp}
 * // Build from scratch
 * auto url = UrlBuilder()
 *     .set_scheme("https")
 *     .set_host("witcher4.com")
 *     .set_path("/ciri")
 *     .set_query("occupation", "witcher")
 *     .build();
 *
 * // Modify existing URL
 * auto modified = UrlBuilder("https://witcher4.com/old/path/to/bestiary")
 *     .set_path("/new/path/to/bestiary")
 *     .set_port("8080")
 *     .build();
 * @endcode
 */
class UrlBuilder {
 private:
  CurlUrlHandle _handle;

  /**
   * @brief Internal helper to set a URL component
   *
   * @param part The URL part to set
   * @param value The value to set. Use `nullptr` to clear
   * @param flags Optional flags for the operation
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the operation fails
   */
  UrlBuilder& set_component(CURLUPart part,
                            char const* value,
                            std::optional<unsigned int> flags = std::nullopt);

 public:
  /**
   * @brief Construct an empty URL builder
   * @exception std::runtime_error if initialization fails
   */
  explicit UrlBuilder();

  /**
   * @brief Construct a URL builder from an existing URL string
   *
   * @param url The URL string to start with
   * @param bitmask_url_flags Optional flags for URL parsing. Common flags include:
   *  - CURLU_DEFAULT_SCHEME: Allows URLs without schemes
   *  - CURLU_NON_SUPPORT_SCHEME: Accept non-supported schemes
   *  - CURLU_URLENCODE: URL encode the path
   * @exception std::runtime_error if the URL cannot be parsed
   */
  explicit UrlBuilder(std::string const& url,
                      std::optional<unsigned int> bitmask_url_flags = std::nullopt);

  /**
   * @brief Construct a URL builder from parsed URL components
   *
   * @param components The parsed URL components to start with
   * @param bitmask_url_flags Optional flags for URL handling
   * @exception std::runtime_error if the components cannot be set
   */
  explicit UrlBuilder(UrlParser::UrlComponents const& components,
                      std::optional<unsigned int> bitmask_url_flags = std::nullopt);

  /**
   * @brief Set the URL scheme (e.g., "http", "https", "ftp")
   *
   * @param scheme The scheme to set. Use `std::nullopt` to clear
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the scheme is invalid
   *
   * Example:
   * @code{.cpp}
   * builder.set_scheme("https");
   * @endcode
   */
  UrlBuilder& set_scheme(std::optional<std::string> const& scheme);

  /**
   * @brief Set the hostname or IP address
   *
   * @param host The host to set. Use `std::nullopt` to clear
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the host is invalid
   *
   * Example:
   * @code{.cpp}
   * builder.set_host("api.example.com");
   * @endcode
   */
  UrlBuilder& set_host(std::optional<std::string> const& host);

  /**
   * @brief Set the port number
   *
   * @param port The port to set as string. Use `std::nullopt` to clear
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the port is invalid
   *
   * Example:
   * @code{.cpp}
   * builder.set_port("8080");
   * @endcode
   */
  UrlBuilder& set_port(std::optional<std::string> const& port);

  /**
   * @brief Set the path component
   *
   * @param path The path to set (should start with "/" for absolute paths). Use `std::nullopt` to
   * clear
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the path is invalid
   *
   * Example:
   * @code{.cpp}
   * builder.set_path("/api/v1/users");
   * @endcode
   */
  UrlBuilder& set_path(std::optional<std::string> const& path);

  /**
   * @brief Set the entire query string
   *
   * @param query The query string (without leading "?"). Use `std::nullopt` to clear
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the query is invalid
   *
   * Example:
   * @code{.cpp}
   * builder.set_query("page=1&limit=10");
   * @endcode
   */
  UrlBuilder& set_query(std::optional<std::string> const& query);

  /**
   * @brief Set the fragment identifier
   *
   * @param fragment The fragment (without leading "#"). Use `std::nullopt` to clear
   * @return Reference to this builder for chaining
   * @exception std::runtime_error if the fragment is invalid
   *
   * Example:
   * @code{.cpp}
   * builder.set_fragment("section-2");
   * @endcode
   */
  UrlBuilder& set_fragment(std::optional<std::string> const& fragment);

  /**
   * @brief Build the final URL string
   *
   * @param bitmask_component_flags Optional flags for URL formatting. Common flags:
   *  - CURLU_PUNYCODE: Convert host to punycode if needed
   *  - CURLU_NO_DEFAULT_PORT: Include port even if it's the default for the scheme
   * @return The complete URL string
   * @exception std::runtime_error if the URL cannot be built
   *
   * Example:
   * @code{.cpp}
   * std::string url = builder.build();
   * @endcode
   */
  std::string build(std::optional<unsigned int> bitmask_component_flags = std::nullopt) const;

  static std::string build_manually(UrlParser::UrlComponents const& components);
};

/**
 * @brief Provides URL encoding functionality
 *
 * The AWS object naming documentation
 * (https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html) lists several
 * types of special characters. In practice, handling them using libcurl is complex and described
 * below.
 *
 *  - Special characters that are safe for use in key names: "!-_.*'()" KvikIO includes !*'() in
 * `aws_special_chars`, because for private bucket they cause AWS authentication by libcurl to fail
 *
 *  - Characters that might require special handling: "&$@=;/:+ ,? and 0-31, 127 ASCII
 * characters". For /, KvikIO does not include it in `aws_special_chars`, because it can be legally
 * used as a path separator. For the space character and ?, although KvikIO has them in
 * `aws_special_chars`, users must manually percent encode them to %20 and %3F, respectively.
 * Otherwise, the space character will be considered malformed by libcurl, and ? cause ambiguity
 * with the query string. For the control characters, KvikIO include them all in
 * `aws_special_chars`.
 *
 *  - Characters to avoid: "\{^}%`]">[~<#| and 128-255 non-ASCII characters". KvikIO recommends
 * users avoiding these characters in the URL. They are not included in `aws_special_chars`.
 *
 */
class UrlEncoder {
 public:
  /**
   * @brief Default set of special characters requiring encoding in AWS URLs
   */
  static constexpr char aws_special_chars[] = {
    '!',    '*',    '\'',   '(',    ')',    '&',    '$',    '@',    '=',    ';',    ':',    '+',
    ' ',    ',',    '?',    '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08',
    '\x09', '\x0A', '\x0B', '\x0C', '\x0D', '\x0E', '\x0F', '\x10', '\x11', '\x12', '\x13', '\x14',
    '\x15', '\x16', '\x17', '\x18', '\x19', '\x1A', '\x1B', '\x1C', '\x1D', '\x1E', '\x1F', '\x7F'};

  /**
   * @brief Percent-encodes specified characters in a URL path
   *
   * Performs percent-encoding (RFC 3986) on a given path string, encoding only the characters
   * specified in the chars_to_encode parameter. Each encoded character is replaced with its
   * percent-encoded equivalent (%XX where XX is the hexadecimal representation of the character).
   *
   * Only ASCII characters (0-127) are supported for encoding. Non-ASCII characters in
   * chars_to_encode will be encoded to an empty string. Characters not in chars_to_encode are
   * passed through unchanged.
   *
   * @param path The path string to encode
   * @param chars_to_encode Set of characters that should be encoded (defaults to aws_special_chars)
   *
   * @return A new string with specified characters percent-encoded
   *
   * @code{.cpp}
   * // Example usage with default AWS special characters
   * std::string encoded = UrlEncoder::encode_path("/path/ with spaces");
   * // Result: "/path/%20with%20spaces"
   *
   * // Example with custom character set
   * std::string encoded = UrlEncoder::encode_path("hello/world", "/");
   * // Result: "hello%2Fworld"
   * @endcode
   */
  static std::string encode_path(std::string_view path,
                                 std::string_view chars_to_encode = std::string_view{
                                   aws_special_chars, sizeof(aws_special_chars)});
};

}  // namespace kvikio::detail
