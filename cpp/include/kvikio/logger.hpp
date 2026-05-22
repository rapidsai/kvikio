/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <string>

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
 *
 * @return Reference to the global KvikIO logger
 */
rapids_logger::logger& default_logger();

/**
 * @brief Returns the global structured-read logger instance for KvikIO.
 *
 * The logger is configured once on first access using the following environment variables:
 *
 * - `KVIKIO_READ_LOG_FILE`: If set, structured read records are written to this file path in
 *   append mode.
 * - `KVIKIO_READ_LOG_LEVEL`: Optional log level (`INFO` by default). Use `OFF` to disable
 *   structured read logging while keeping `KVIKIO_READ_LOG_FILE` set.
 * - `KVIKIO_READ_LOG_REDACT_QUERY`: Optional boolean (`ON` by default). When enabled, URL query
 *   strings are removed from the `source` field.
 *
 * @return Reference to the global structured-read logger.
 */
rapids_logger::logger& read_logger();

/**
 * @brief Whether structured read logging is enabled.
 */
bool is_read_logging_enabled();

/**
 * @brief Return a URL string sanitized for structured read logging.
 */
std::string sanitize_read_log_url(std::string const& url);

/**
 * @brief Emit one structured read record as NDJSON.
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
 */
void log_structured_read(std::string const& source,
                         std::int64_t start,
                         std::int64_t end,
                         std::size_t offset,
                         std::size_t size,
                         std::size_t bytes_read,
                         char const* backend,
                         char const* status,
                         bool is_device_buffer,
                         std::size_t request_id);
}  // namespace KVIKIO_EXPORT kvikio
