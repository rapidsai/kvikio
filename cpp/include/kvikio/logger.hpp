/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

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
}  // namespace KVIKIO_EXPORT kvikio
