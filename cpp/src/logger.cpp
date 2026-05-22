/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <kvikio/logger.hpp>

namespace KVIKIO_EXPORT kvikio {

namespace {
rapids_logger::level_enum parse_level(char const* env)
{
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

rapids_logger::level_enum get_default_level_from_env()
{
  return parse_level(std::getenv("KVIKIO_LOG_LEVEL"));
}

rapids_logger::level_enum get_read_level_from_env()
{
  auto const* env = std::getenv("KVIKIO_READ_LOG_LEVEL");
  if (env == nullptr) { return rapids_logger::level_enum::info; }
  return parse_level(env);
}

bool parse_bool_env(char const* env, bool default_val)
{
  if (env == nullptr) { return default_val; }
  std::string val{env};
  std::transform(
    val.begin(), val.end(), val.begin(), [](unsigned char c) { return std::tolower(c); });
  if (val == "1" || val == "true" || val == "on" || val == "yes") { return true; }
  if (val == "0" || val == "false" || val == "off" || val == "no") { return false; }
  return default_val;
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

rapids_logger::sink_ptr make_read_sink(rapids_logger::level_enum level)
{
  if (level == rapids_logger::level_enum::off) {
    return std::make_shared<rapids_logger::null_sink_mt>();
  }
  auto const* path = std::getenv("KVIKIO_READ_LOG_FILE");
  if (path == nullptr || path[0] == '\0') {
    return std::make_shared<rapids_logger::null_sink_mt>();
  }
  if (std::string_view{path} == "-") {
    return std::make_shared<rapids_logger::ostream_sink_mt>(std::cout, true);
  }
  try {
    bool const truncate{false};  // Append to existing file for post-hoc analysis.
    return std::make_shared<rapids_logger::basic_file_sink_mt>(path, truncate);
  } catch (std::exception const& e) {
    std::cerr << "KvikIO warning: Cannot open read log file " << path << ": " << e.what()
              << ". Structured read logging is disabled.\n";
    return std::make_shared<rapids_logger::null_sink_mt>();
  }
}

std::string escape_json(std::string const& input)
{
  std::string out;
  out.reserve(input.size());
  for (auto const c : input) {
    switch (c) {
      case '\"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::ostringstream oss;
          oss << "\\u" << std::hex << std::uppercase << std::setfill('0') << std::setw(4)
              << static_cast<int>(static_cast<unsigned char>(c));
          out += oss.str();
        } else {
          out += c;
        }
    }
  }
  return out;
}
}  // namespace

rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    auto const level = get_default_level_from_env();
    rapids_logger::logger logger_{"kvikio", {make_sink(level)}};
    // Pattern: [thread_id][hours:minutes:seconds:microseconds][level ] message
    logger_.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    logger_.set_level(level);
    return logger_;
  }();
  return logger_;
}

rapids_logger::logger& read_logger()
{
  static rapids_logger::logger logger_ = [] {
    auto const level = get_read_level_from_env();
    rapids_logger::logger logger_{"kvikio_read", {make_read_sink(level)}};
    logger_.set_pattern("%v");
    logger_.set_level(level);
    return logger_;
  }();
  return logger_;
}

bool is_read_logging_enabled()
{
  auto const* path = std::getenv("KVIKIO_READ_LOG_FILE");
  if (path == nullptr || path[0] == '\0') { return false; }
  return read_logger().level() != rapids_logger::level_enum::off;
}

std::string sanitize_read_log_url(std::string const& url)
{
  auto const redact_query = parse_bool_env(std::getenv("KVIKIO_READ_LOG_REDACT_QUERY"), true);
  if (!redact_query) { return url; }
  auto const pos = url.find('?');
  return pos == std::string::npos ? url : url.substr(0, pos);
}

void log_structured_read(std::string const& source,
                         std::int64_t start,
                         std::int64_t end,
                         std::size_t offset,
                         std::size_t size,
                         std::size_t bytes_read,
                         char const* backend,
                         char const* status,
                         bool is_device_buffer,
                         std::size_t request_id)
{
  if (!is_read_logging_enabled()) { return; }
  auto const sanitized = sanitize_read_log_url(source);
  auto const thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto json            = std::string{};
  json.reserve(sanitized.size() + 256);
  json += "{\"source\":\"";
  json += escape_json(sanitized);
  json += "\",\"start\":";
  json += std::to_string(start);
  json += ",\"end\":";
  json += std::to_string(end);
  json += ",\"offset\":";
  json += std::to_string(offset);
  json += ",\"size\":";
  json += std::to_string(size);
  json += ",\"threadId\":";
  json += std::to_string(thread_id);
  json += ",\"bytesRead\":";
  json += std::to_string(bytes_read);
  json += ",\"backend\":\"";
  json += escape_json(backend);
  json += "\",\"status\":\"";
  json += escape_json(status);
  json += "\",\"isDeviceBuffer\":";
  json += is_device_buffer ? "true" : "false";
  json += ",\"requestId\":";
  json += std::to_string(request_id);
  json += "}";
  read_logger().log(rapids_logger::level_enum::info, json);
}
}  // namespace KVIKIO_EXPORT kvikio
