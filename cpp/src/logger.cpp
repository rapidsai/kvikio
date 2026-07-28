/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <kvikio/logger.hpp>

namespace KVIKIO_EXPORT kvikio {

namespace {
enum class LogFormat { TEXT, JSON };

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

LogFormat get_log_format_from_env()
{
  auto const* env = std::getenv("KVIKIO_LOG_FORMAT");
  if (env == nullptr) { return LogFormat::TEXT; }
  std::string value{env};
  std::transform(
    value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  return value == "json" ? LogFormat::JSON : LogFormat::TEXT;
}

LogFormat log_format()
{
  static auto const format = get_log_format_from_env();
  return format;
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

std::int64_t epoch_nanos()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

char const* level_name(rapids_logger::level_enum level)
{
  switch (level) {
    case rapids_logger::level_enum::trace: return "trace";
    case rapids_logger::level_enum::debug: return "debug";
    case rapids_logger::level_enum::info: return "info";
    case rapids_logger::level_enum::warn: return "warn";
    case rapids_logger::level_enum::error: return "error";
    case rapids_logger::level_enum::critical: return "critical";
    case rapids_logger::level_enum::off: return "off";
    default: return "unknown";
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

std::size_t thread_id() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }
}  // namespace

rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    auto const level = get_default_level_from_env();
    rapids_logger::logger logger_{"kvikio", {make_sink(level)}};
    if (log_format() == LogFormat::JSON) {
      // JSON serialization is handled before messages reach rapids-logger.
      logger_.set_pattern("%v");
    } else {
      // Pattern: [thread_id][hours:minutes:seconds:microseconds][level ] message
      logger_.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    }
    logger_.set_level(level);
    return logger_;
  }();
  return logger_;
}

Logger& formatted_logger()
{
  static Logger logger_;
  return logger_;
}

void log_message(rapids_logger::level_enum level, std::string const& message)
{
  if (log_format() == LogFormat::TEXT) {
    default_logger().log(level, message);
    return;
  }

  std::string json;
  json.reserve(message.size() + 128);
  json += "{\"event\":\"log\",\"timestamp\":";
  json += std::to_string(epoch_nanos());
  json += ",\"threadId\":";
  json += std::to_string(thread_id());
  json += ",\"level\":\"";
  json += level_name(level);
  json += "\",\"message\":\"";
  json += escape_json(message);
  json += "\"}";
  default_logger().log(level, json);
}

std::string sanitize_read_log_url(std::string const& url)
{
  auto const redact_query = parse_bool_env(std::getenv("KVIKIO_LOG_REDACT_QUERY"), true);
  if (!redact_query) { return url; }
  auto const pos = url.find('?');
  return pos == std::string::npos ? url : url.substr(0, pos);
}

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
                       char const* method)
{
  if (!default_logger().should_log(rapids_logger::level_enum::trace)) { return; }
  auto const sanitized         = sanitize_read_log_url(source);
  auto const current_thread_id = thread_id();
  if (log_format() == LogFormat::TEXT) {
    if (method == nullptr) {
      default_logger().log(
        rapids_logger::level_enum::trace,
        "read(source=%s, start=%lld, end=%lld, offset=%zu, size=%zu, thread_id=%zu, "
        "bytes_read=%zu, backend=%s, status=%s, is_device_buffer=%s, request_id=%zu)",
        sanitized,
        static_cast<long long>(start),
        static_cast<long long>(end),
        offset,
        size,
        current_thread_id,
        bytes_read,
        backend,
        status,
        is_device_buffer ? "true" : "false",
        request_id);
    } else {
      default_logger().log(
        rapids_logger::level_enum::trace,
        "read(source=%s, start=%lld, end=%lld, offset=%zu, size=%zu, thread_id=%zu, "
        "bytes_read=%zu, backend=%s, status=%s, is_device_buffer=%s, request_id=%zu, "
        "method=%s)",
        sanitized,
        static_cast<long long>(start),
        static_cast<long long>(end),
        offset,
        size,
        current_thread_id,
        bytes_read,
        backend,
        status,
        is_device_buffer ? "true" : "false",
        request_id,
        method);
    }
    return;
  }

  auto json = std::string{};
  json.reserve(sanitized.size() + 280);
  json += "{\"event\":\"read\",\"level\":\"trace\",\"source\":\"";
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
  json += std::to_string(current_thread_id);
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
  if (method != nullptr) {
    json += ",\"method\":\"";
    json += escape_json(method);
    json += "\"";
  }
  json += "}";
  default_logger().log(rapids_logger::level_enum::trace, json);
}

void log_http_request(std::string const& source,
                      std::int64_t start,
                      std::int64_t end,
                      char const* method,
                      char const* status,
                      char const* purpose)
{
  if (!default_logger().should_log(rapids_logger::level_enum::trace)) { return; }
  auto const sanitized         = sanitize_read_log_url(source);
  auto const current_thread_id = thread_id();
  if (log_format() == LogFormat::TEXT) {
    default_logger().log(
      rapids_logger::level_enum::trace,
      "http(source=%s, start=%lld, end=%lld, thread_id=%zu, method=%s, status=%s, purpose=%s)",
      sanitized,
      static_cast<long long>(start),
      static_cast<long long>(end),
      current_thread_id,
      method,
      status,
      purpose);
    return;
  }

  auto json = std::string{};
  json.reserve(sanitized.size() + 192);
  json += "{\"event\":\"http\",\"level\":\"trace\",\"source\":\"";
  json += escape_json(sanitized);
  json += "\",\"start\":";
  json += std::to_string(start);
  json += ",\"end\":";
  json += std::to_string(end);
  json += ",\"threadId\":";
  json += std::to_string(current_thread_id);
  json += ",\"method\":\"";
  json += escape_json(method);
  json += "\",\"status\":\"";
  json += escape_json(status);
  json += "\",\"purpose\":\"";
  json += escape_json(purpose);
  json += "\"}";
  default_logger().log(rapids_logger::level_enum::trace, json);
}
}  // namespace KVIKIO_EXPORT kvikio
