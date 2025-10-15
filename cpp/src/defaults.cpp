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

#include <cstddef>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <BS_thread_pool.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/http_status_codes.hpp>
#include <kvikio/shim/cufile.hpp>
#include <string_view>

namespace kvikio {
template <>
bool getenv_or(std::string_view env_var_name, bool default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  try {
    // Try parsing `env_var_name` as a integer
    return static_cast<bool>(std::stoi(env_val));
  } catch (std::invalid_argument const&) {
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
  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" + std::string{env_val},
              std::invalid_argument);
  return {};
}

template <>
CompatMode getenv_or(std::string_view env_var_name, CompatMode default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  return detail::parse_compat_mode_str(env_val);
}

template <>
std::vector<int> getenv_or(std::string_view env_var_name, std::vector<int> default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto* const env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return std::move(default_val); }
  std::string const int_str(env_val);
  if (int_str.empty()) { return std::move(default_val); }

  return detail::parse_http_status_codes(env_var_name, int_str);
}

unsigned int defaults::get_num_threads_from_env()
{
  KVIKIO_NVTX_FUNC_RANGE();

  auto const [env_var_name, num_threads, _] =
    getenv_or({"KVIKIO_NTHREADS", "KVIKIO_NUM_THREADS"}, 1);
  KVIKIO_EXPECT(num_threads > 0,
                std::string{env_var_name} + " has to be a positive integer",
                std::invalid_argument);
  return num_threads;
}

defaults::defaults()
{
  KVIKIO_NVTX_FUNC_RANGE();
  // Determine the default value of `compat_mode`
  {
    _compat_mode = getenv_or("KVIKIO_COMPAT_MODE", CompatMode::AUTO);
  }
  // Determine the default value of `task_size`
  {
    ssize_t const env = getenv_or("KVIKIO_TASK_SIZE", 4 * 1024 * 1024);
    KVIKIO_EXPECT(env > 0, "KVIKIO_TASK_SIZE has to be a positive integer", std::invalid_argument);
    _task_size = env;
  }
  // Determine the default value of `gds_threshold`
  {
    ssize_t const env = getenv_or("KVIKIO_GDS_THRESHOLD", 16 * 1024);
    KVIKIO_EXPECT(
      env >= 0, "KVIKIO_GDS_THRESHOLD has to be a positive integer", std::invalid_argument);
    _gds_threshold = env;
  }
  // Determine the default value of `bounce_buffer_size`
  {
    ssize_t const env = getenv_or("KVIKIO_BOUNCE_BUFFER_SIZE", 16 * 1024 * 1024);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_BOUNCE_BUFFER_SIZE has to be a positive integer", std::invalid_argument);
    _bounce_buffer_size = env;
  }
  // Determine the default value of `http_max_attempts`
  {
    ssize_t const env = getenv_or("KVIKIO_HTTP_MAX_ATTEMPTS", 3);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_HTTP_MAX_ATTEMPTS has to be a positive integer", std::invalid_argument);
    _http_max_attempts = env;
  }

  // Determine the default value of `http_timeout`
  {
    long const env = getenv_or("KVIKIO_HTTP_TIMEOUT", 60);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_HTTP_TIMEOUT has to be a positive integer", std::invalid_argument);
    _http_timeout = env;
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

void defaults::set_compat_mode(CompatMode compat_mode) { instance()->_compat_mode = compat_mode; }

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

BS_thread_pool& defaults::thread_pool() { return instance()->_thread_pool; }

unsigned int defaults::thread_pool_nthreads() { return thread_pool().get_thread_count(); }

void defaults::set_thread_pool_nthreads(unsigned int nthreads)
{
  KVIKIO_EXPECT(
    nthreads > 0, "number of threads must be a positive integer", std::invalid_argument);
  thread_pool().reset(nthreads);
}

unsigned int defaults::num_threads() { return thread_pool_nthreads(); }

void defaults::set_num_threads(unsigned int nthreads) { set_thread_pool_nthreads(nthreads); }

std::size_t defaults::task_size() { return instance()->_task_size; }

void defaults::set_task_size(std::size_t nbytes)
{
  KVIKIO_EXPECT(nbytes > 0, "task size must be a positive integer", std::invalid_argument);
  instance()->_task_size = nbytes;
}

std::size_t defaults::gds_threshold() { return instance()->_gds_threshold; }

void defaults::set_gds_threshold(std::size_t nbytes) { instance()->_gds_threshold = nbytes; }

std::size_t defaults::bounce_buffer_size() { return instance()->_bounce_buffer_size; }

void defaults::set_bounce_buffer_size(std::size_t nbytes)
{
  KVIKIO_EXPECT(
    nbytes > 0, "size of the bounce buffer must be a positive integer", std::invalid_argument);
  instance()->_bounce_buffer_size = nbytes;
}

std::size_t defaults::http_max_attempts() { return instance()->_http_max_attempts; }

void defaults::set_http_max_attempts(std::size_t attempts)
{
  KVIKIO_EXPECT(attempts > 0, "attempts must be a positive integer", std::invalid_argument);
  instance()->_http_max_attempts = attempts;
}

std::vector<int> const& defaults::http_status_codes() { return instance()->_http_status_codes; }

void defaults::set_http_status_codes(std::vector<int> status_codes)
{
  instance()->_http_status_codes = std::move(status_codes);
}

long defaults::http_timeout() { return instance()->_http_timeout; }
void defaults::set_http_timeout(long timeout_seconds)
{
  KVIKIO_EXPECT(
    timeout_seconds > 0, "timeout_seconds must be a positive integer", std::invalid_argument);
  instance()->_http_timeout = timeout_seconds;
}

}  // namespace kvikio
