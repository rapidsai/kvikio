/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <iostream>

#include <kvikio/logger.hpp>

namespace KVIKIO_EXPORT kvikio {

namespace {
rapids_logger::level_enum get_level_from_env()
{
  auto const* env = std::getenv("KVIKIO_LOG_LEVEL");
  if (env == nullptr) { return rapids_logger::level_enum::off; }

  // Convert to lowercase
  std::string val{env};
  std::transform(
    val.begin(), val.end(), val.begin(), [](unsigned char c) { return std::tolower(c); });

  if (val == "trace") return rapids_logger::level_enum::trace;
  if (val == "debug") return rapids_logger::level_enum::debug;
  if (val == "info") return rapids_logger::level_enum::info;
  if (val == "warn") return rapids_logger::level_enum::warn;
  if (val == "error") return rapids_logger::level_enum::error;
  if (val == "critical") return rapids_logger::level_enum::critical;
  if (val == "off") return rapids_logger::level_enum::off;

  // Ignore invalid log value
  return rapids_logger::level_enum::off;
}

rapids_logger::sink_ptr make_sink(rapids_logger::level_enum level)
{
  if (level == rapids_logger::level_enum::off) {
    return std::make_shared<rapids_logger::null_sink_mt>();
  }

  auto const* path = std::getenv("KVIKIO_LOG_FILE");
  if (path == nullptr) { return std::make_shared<rapids_logger::stderr_sink_mt>(); }

  try {
    bool const truncate{true};  // Clear the file when the sink opens it
    return std::make_shared<rapids_logger::basic_file_sink_mt>(path, truncate);
  } catch (std::exception const& e) {
    std::cerr << "KvikIO warning: Cannot open log file " << path << ": " << e.what()
              << ". Logging to the standard error instead\n";
    return std::make_shared<rapids_logger::stderr_sink_mt>();
  }
}
}  // namespace

rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    auto const level = get_level_from_env();
    rapids_logger::logger logger_{"kvikio", {make_sink(level)}};
    // Pattern: [thread_id][hours:minutes:seconds:microseconds][level ] message
    logger_.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    logger_.set_level(level);
    return logger_;
  }();
  return logger_;
}
}  // namespace KVIKIO_EXPORT kvikio
