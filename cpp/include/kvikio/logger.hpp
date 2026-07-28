/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <kvikio/logger_macros.hpp>
#include <kvikio/shim/utils.hpp>

#include <rapids_logger/logger.hpp>

namespace KVIKIO_EXPORT kvikio {
/**
 * @brief Returns the global logger instance for KvikIO.
 *
 * The logger is configured once on first access using the following environment variables:
 *
 * - `KVIKIO_LOG_LEVEL`: Sets the log level. Accepted values (case-insensitive) are `TRACE`,
 * `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`, and `OFF`. If unset or set to any other value,
 * logging is disabled.
 * - `KVIKIO_LOG_FILE`: If set, log output is written to this file path (overwritten on each process
 * start). If the file cannot be opened, falls back to stderr with a warning. Has no effect when
 * logging is disabled.
 * - `KVIKIO_LOG_FORMAT`: Sets the output format to `TEXT` (the default) or `JSON`.
 *
 * @return Reference to the global KvikIO logger
 */
rapids_logger::logger& default_logger();

/**
 * @brief Log a preformatted message using the configured output format.
 */
void log_message(rapids_logger::level_enum level, std::string const& message);

/**
 * @brief Logger facade that applies KvikIO's configured output format.
 */
class Logger {
 public:
  template <typename... Args>
  void log(rapids_logger::level_enum level, std::string const& format, Args&&... args)
  {
    if (!default_logger().should_log(level)) { return; }

    auto convert_to_c_string = [](auto&& arg) -> decltype(auto) {
      using ArgType = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<ArgType, std::string>) {
        return arg.c_str();
      } else {
        return std::forward<decltype(arg)>(arg);
      }
    };

    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
    auto const formatted_size =
      std::snprintf(nullptr, 0, format.c_str(), convert_to_c_string(std::forward<Args>(args))...);
    if (formatted_size < 0) { throw std::runtime_error("Error during formatting."); }
    if (formatted_size == 0) {
      log_message(level, {});
      return;
    }
    auto const size = static_cast<std::size_t>(formatted_size) + 1;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays, cppcoreguidelines-avoid-c-arrays)
    auto buffer = std::make_unique<char[]>(size);
    std::snprintf(
      buffer.get(), size, format.c_str(), convert_to_c_string(std::forward<Args>(args))...);
    // NOLINTEND(cppcoreguidelines-pro-type-vararg)
    log_message(level, {buffer.get(), buffer.get() + formatted_size});
  }
};

/**
 * @brief Returns the global output-formatting logger facade.
 */
Logger& formatted_logger();

/**
 * @brief Return a URL string sanitized for read logging.
 */
std::string sanitize_read_log_url(std::string const& url);

/**
 * @brief Emit one physical-read record at TRACE level.
 *
 * Record schema:
 * - `source: str`
 * - `start: int` (epoch nanoseconds)
 * - `end: int` (epoch nanoseconds)
 * - `offset: int`
 * - `size: int`
 * - `threadId: int`
 * - `bytesRead: int`
 * - `backend: str`
 * - `status: str`
 * - `isDeviceBuffer: bool`
 * - `requestId: int`
 * - `method: str` (optional; HTTP method such as `"GET"` for remote reads)
 */
void log_physical_read(std::string const& source,
                       std::int64_t start,
                       std::int64_t end,
                       std::size_t offset,
                       std::size_t size,
                       std::size_t bytes_read,
                       char const* backend,
                       char const* status,
                       bool is_device_buffer,
                       std::size_t request_id,
                       char const* method = nullptr);

/**
 * @brief Emit one HTTP metadata / probe request record at TRACE level.
 *
 * Used for size discovery and similar non-data-read HTTP operations (for example HEAD
 * during `RemoteFile` open, or a one-byte GET used when HEAD is unavailable).
 *
 * Record schema:
 * - `event: "http"`
 * - `source: str`
 * - `start: int` (epoch nanoseconds)
 * - `end: int` (epoch nanoseconds)
 * - `threadId: int`
 * - `method: str` (for example `"HEAD"` or `"GET"`)
 * - `status: str`
 * - `purpose: str` (for example `"metadata"`)
 */
void log_http_request(std::string const& source,
                      std::int64_t start,
                      std::int64_t end,
                      char const* method,
                      char const* status,
                      char const* purpose = "metadata");
}  // namespace KVIKIO_EXPORT kvikio
