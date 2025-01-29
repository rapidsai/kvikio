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

}  // namespace detail

template <>
bool getenv_or(std::string_view env_var_name, bool default_val)
{
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

unsigned int defaults::get_num_threads_from_env()
{
  int const ret = getenv_or("KVIKIO_NTHREADS", 1);
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
    ssize_t const env = getenv_or("KVIKIO_TASK_SIZE", 4 * 1024 * 1024);
    if (env <= 0) {
      throw std::invalid_argument(
        "KVIKIO_TASK_SIZE has to be a positive integer greater than zero");
    }
    _task_size = env;
  }
  // Determine the default value of `gds_threshold`
  {
    ssize_t const env = getenv_or("KVIKIO_GDS_THRESHOLD", 1024 * 1024);
    if (env < 0) {
      throw std::invalid_argument("KVIKIO_GDS_THRESHOLD has to be a positive integer");
    }
    _gds_threshold = env;
  }
  // Determine the default value of `bounce_buffer_size`
  {
    ssize_t const env = getenv_or("KVIKIO_BOUNCE_BUFFER_SIZE", 16 * 1024 * 1024);
    if (env <= 0) {
      throw std::invalid_argument(
        "KVIKIO_BOUNCE_BUFFER_SIZE has to be a positive integer greater than zero");
    }
    _bounce_buffer_size = env;
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

BS_thread_pool& defaults::thread_pool() { return instance()->_thread_pool; }

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

}  // namespace kvikio
