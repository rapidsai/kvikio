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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <BS_thread_pool.hpp>

#include <kvikio/defaults.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

namespace detail {
CompatMode parse_compat_mode_str(std::string_view compat_mode_str)
{
  // Convert to lowercase
  std::string tmp{compat_mode_str};
  std::transform(
    tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });

  CompatMode res{};
  if (tmp == "on" || tmp == "true" || tmp == "yes" || tmp == "1") {
    res = CompatMode::ON;
  } else if (tmp == "off" || tmp == "false" || tmp == "no" || tmp == "0") {
    res = CompatMode::OFF;
  } else if (tmp == "auto") {
    res = CompatMode::AUTO;
  } else {
    throw std::invalid_argument("Unknown compatibility mode: " + std::string{tmp});
  }
  return res;
}

std::vector<int> parse_http_status_codes(std::string_view env_var_name,
                                         std::string const status_codes)
{
  // Ensure `status_codes` consists only of 3-digit integers separated by commas, allowing spaces.
  std::regex const check_pattern(R"(^\s*\d{3}\s*(\s*,\s*\d{3}\s*)*$)");
  if (!std::regex_match(status_codes, check_pattern)) {
    throw std::invalid_argument(std::string{env_var_name} +
                                ": invalid format, expected comma-separated integers.");
  }

  // Match every integer in `status_codes`.
  std::regex const number_pattern(R"(\d+)");

  // For each match, we push_back `std::stoi(match.str())` into `ret`.
  std::vector<int> ret;
  std::transform(std::sregex_iterator(status_codes.begin(), status_codes.end(), number_pattern),
                 std::sregex_iterator(),
                 std::back_inserter(ret),
                 [](std::smatch const& match) -> int { return std::stoi(match.str()); });
  return ret;
}

}  // namespace detail

template <>
bool getenv_or(std::string_view env_var_name, bool default_val)
{
  const auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  try {
    // Try parsing `env_var_name` as a integer
    return static_cast<bool>(std::stoi(env_val));
  } catch (const std::invalid_argument&) {
  }
  // Convert to lowercase
  std::string str{env_val};
  // Special considerations regarding the case conversion:
  // - std::tolower() is not an addressable function. Passing it to std::transform() as
  //   a function pointer, if the compile turns out successful, causes the program behavior
  //   "unspecified (possibly ill-formed)", hence the lambda. ::tolower() is addressable
  //   and does not have this problem, but the following item still applies.
  // - To avoid UB in std::tolower() or ::tolower(), the character must be cast to unsigned char.
  std::transform(
    str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  // Trim whitespaces
  std::stringstream trimmer;
  trimmer << str;
  str.clear();
  trimmer >> str;
  // Match value
  if (str == "true" || str == "on" || str == "yes") { return true; }
  if (str == "false" || str == "off" || str == "no") { return false; }
  throw std::invalid_argument("unknown config value " + std::string{env_var_name} + "=" +
                              std::string{env_val});
}

template <>
CompatMode getenv_or(std::string_view env_var_name, CompatMode default_val)
{
  auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  return detail::parse_compat_mode_str(env_val);
}

template <>
std::vector<int> getenv_or(std::string_view env_var_name, std::vector<int> default_val)
{
  auto* const env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  std::string const int_str(env_val);
  if (int_str.empty()) { return default_val; }

  return detail::parse_http_status_codes(env_var_name, int_str);
}

unsigned int defaults::get_num_threads_from_env()
{
  const int ret = getenv_or("KVIKIO_NTHREADS", 1);
  if (ret <= 0) {
    throw std::invalid_argument("KVIKIO_NTHREADS has to be a positive integer greater than zero");
  }
  return ret;
}

defaults::defaults()
{
  // Determine the default value of `compat_mode`
  {
    _compat_mode = getenv_or("KVIKIO_COMPAT_MODE", CompatMode::AUTO);
  }
  // Determine the default value of `task_size`
  {
    const ssize_t env = getenv_or("KVIKIO_TASK_SIZE", 4 * 1024 * 1024);
    if (env <= 0) {
      throw std::invalid_argument(
        "KVIKIO_TASK_SIZE has to be a positive integer greater than zero");
    }
    _task_size = env;
  }
  // Determine the default value of `gds_threshold`
  {
    const ssize_t env = getenv_or("KVIKIO_GDS_THRESHOLD", 1024 * 1024);
    if (env < 0) {
      throw std::invalid_argument("KVIKIO_GDS_THRESHOLD has to be a positive integer");
    }
    _gds_threshold = env;
  }
  // Determine the default value of `bounce_buffer_size`
  {
    const ssize_t env = getenv_or("KVIKIO_BOUNCE_BUFFER_SIZE", 16 * 1024 * 1024);
    if (env <= 0) {
      throw std::invalid_argument(
        "KVIKIO_BOUNCE_BUFFER_SIZE has to be a positive integer greater than zero");
    }
    _bounce_buffer_size = env;
  }
  // Determine the default value of `max_attempts`
  {
    const ssize_t env = getenv_or("KVIKIO_MAX_ATTEMPTS", 3);
    if (env <= 0) {
      throw std::invalid_argument("KVIKIO_MAX_ATTEMPTS has to be a positive integer");
    }
    _max_attempts = env;
  }
  // Determine the default value of `http_status_codes`
  {
    _http_status_codes =
      getenv_or("KVIKIO_HTTP_STATUS_CODES", std::vector<int>{429, 500, 502, 503, 504});
  }
}

defaults* defaults::instance()
{
  static defaults _instance;
  return &_instance;
}
CompatMode defaults::compat_mode() { return instance()->_compat_mode; }

void defaults::compat_mode_reset(CompatMode compat_mode) { instance()->_compat_mode = compat_mode; }

CompatMode defaults::infer_compat_mode_if_auto(CompatMode compat_mode) noexcept
{
  if (compat_mode == CompatMode::AUTO) {
    static auto inferred_compat_mode_for_auto = []() -> CompatMode {
      return is_cufile_available() ? CompatMode::OFF : CompatMode::ON;
    }();
    return inferred_compat_mode_for_auto;
  }
  return compat_mode;
}

bool defaults::is_compat_mode_preferred(CompatMode compat_mode) noexcept
{
  return compat_mode == CompatMode::ON ||
         (compat_mode == CompatMode::AUTO &&
          defaults::infer_compat_mode_if_auto(compat_mode) == CompatMode::ON);
}

bool defaults::is_compat_mode_preferred() { return is_compat_mode_preferred(compat_mode()); }

BS::thread_pool& defaults::thread_pool() { return instance()->_thread_pool; }

unsigned int defaults::thread_pool_nthreads() { return thread_pool().get_thread_count(); }

void defaults::thread_pool_nthreads_reset(unsigned int nthreads)
{
  if (nthreads == 0) {
    throw std::invalid_argument("number of threads must be a positive integer greater than zero");
  }
  thread_pool().reset(nthreads);
}

std::size_t defaults::task_size() { return instance()->_task_size; }

void defaults::task_size_reset(std::size_t nbytes)
{
  if (nbytes == 0) {
    throw std::invalid_argument("task size must be a positive integer greater than zero");
  }
  instance()->_task_size = nbytes;
}

std::size_t defaults::gds_threshold() { return instance()->_gds_threshold; }

void defaults::gds_threshold_reset(std::size_t nbytes) { instance()->_gds_threshold = nbytes; }

std::size_t defaults::bounce_buffer_size() { return instance()->_bounce_buffer_size; }

void defaults::bounce_buffer_size_reset(std::size_t nbytes)
{
  if (nbytes == 0) {
    throw std::invalid_argument(
      "size of the bounce buffer must be a positive integer greater than zero");
  }
  instance()->_bounce_buffer_size = nbytes;
}

std::size_t defaults::max_attempts() { return instance()->_max_attempts; }

void defaults::max_attempts_reset(std::size_t attempts)
{
  if (attempts == 0) { throw std::invalid_argument("max_attempts must be a positive integer"); }
  instance()->_max_attempts = attempts;
}

std::vector<int> const& defaults::http_status_codes() { return instance()->_http_status_codes; }

void defaults::http_status_codes_reset(std::vector<int> status_codes)
{
  instance()->_http_status_codes = status_codes;
}

}  // namespace kvikio
