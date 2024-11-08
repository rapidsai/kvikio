/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include <BS_thread_pool.hpp>

#include <kvikio/shim/cufile.hpp>

namespace kvikio {

/**
 * @brief I/O compatibility mode.
 */
enum class CompatMode : uint8_t {
  OFF,    // Enforce cuFile I/O. Undefined behavior for KvikIO if this option is selected but the
          // system config check does not pass: The program may error out, crash or hang on I/O
          // operations.
  ON,     // Enforce POSIX I/O.
  ALLOW,  // Use cuFile I/O. Fall back to the POSIX I/O if the system config check does not
          // pass.
};

namespace detail {
template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  const auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  if (sstream.fail()) {
    throw std::invalid_argument("unknown config value " + std::string{env_var_name} + "=" +
                                std::string{env_val});
  }
  return converted_val;
}

template <>
inline bool getenv_or(std::string_view env_var_name, bool default_val)
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
inline CompatMode getenv_or(std::string_view env_var_name, CompatMode default_val)
{
  auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  // Convert to lowercase
  std::string env_val_lowercase{env_val};
  std::transform(env_val_lowercase.begin(),
                 env_val_lowercase.end(),
                 env_val_lowercase.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  CompatMode res{};
  if (env_val_lowercase == "on") {
    res = CompatMode::ON;
  } else if (env_val_lowercase == "off") {
    res = CompatMode::OFF;
  } else if (env_val_lowercase == "allow") {
    res = CompatMode::ALLOW;
  } else {
    throw std::invalid_argument("unknown config value " + std::string{env_var_name} + "=" +
                                std::string{env_val});
  }

  return res;
}

}  // namespace detail

/**
 * @brief Singleton class of default values used throughout KvikIO.
 *
 */
class defaults {
 private:
  BS::thread_pool _thread_pool{get_num_threads_from_env()};
  CompatMode _compat_mode;
  std::size_t _task_size;
  std::size_t _gds_threshold;
  std::size_t _bounce_buffer_size;

  static unsigned int get_num_threads_from_env()
  {
    const int ret = detail::getenv_or("KVIKIO_NTHREADS", 1);
    if (ret <= 0) {
      throw std::invalid_argument("KVIKIO_NTHREADS has to be a positive integer greater than zero");
    }
    return ret;
  }

  /**
   * @brief Set the internal compatibility mode to either ON or OFF according to the actual system
   * config check. While users can set the compatibility mode to one of the three options, this
   * function reduces the internal state to two possibilities in order to determine the actual I/O
   * path.
   */
  void readjust_compat_mode()
  {
    if (is_cufile_available()) {
      _compat_mode = CompatMode::OFF;
    } else {
      _compat_mode = CompatMode::ON;
    }
  }

  defaults()
  {
    // Determine the default value of `compat_mode`
    {
      _compat_mode = detail::getenv_or("KVIKIO_COMPAT_MODE", CompatMode::ALLOW);
      if (_compat_mode == CompatMode::ALLOW) {
        // Infer based on runtime environment
        readjust_compat_mode();
      }
    }
    // Determine the default value of `task_size`
    {
      const ssize_t env = detail::getenv_or("KVIKIO_TASK_SIZE", 4 * 1024 * 1024);
      if (env <= 0) {
        throw std::invalid_argument(
          "KVIKIO_TASK_SIZE has to be a positive integer greater than zero");
      }
      _task_size = env;
    }
    // Determine the default value of `gds_threshold`
    {
      const ssize_t env = detail::getenv_or("KVIKIO_GDS_THRESHOLD", 1024 * 1024);
      if (env < 0) {
        throw std::invalid_argument("KVIKIO_GDS_THRESHOLD has to be a positive integer");
      }
      _gds_threshold = env;
    }
    // Determine the default value of `bounce_buffer_size`
    {
      const ssize_t env = detail::getenv_or("KVIKIO_BOUNCE_BUFFER_SIZE", 16 * 1024 * 1024);
      if (env <= 0) {
        throw std::invalid_argument(
          "KVIKIO_BOUNCE_BUFFER_SIZE has to be a positive integer greater than zero");
      }
      _bounce_buffer_size = env;
    }
  }

  KVIKIO_EXPORT static defaults* instance()
  {
    static defaults _instance;
    return &_instance;
  }

 public:
  /**
   * @brief Return whether the KvikIO library is running in compatibility mode or not
   *
   * Notice, this is not the same as the compatibility mode in cuFile. That is,
   * cuFile can run in compatibility mode while KvikIO is not.
   *
   * When KvikIO is running in compatibility mode, it doesn't load `libcufile.so`. Instead,
   * reads and writes are done using POSIX.
   *
   * Set the environment variable `KVIKIO_COMPAT_MODE` to enable/disable compatibility mode.
   * By default, compatibility mode is enabled:
   *  - when `libcufile` cannot be found
   *  - when running in Windows Subsystem for Linux (WSL)
   *  - when `/run/udev` isn't readable, which typically happens when running inside a docker
   *    image not launched with `--volume /run/udev:/run/udev:ro`
   *
   * @return The boolean answer
   */
  [[nodiscard]] static bool compat_mode() { return instance()->_compat_mode == CompatMode::ON; }

  /**
   * @brief Reset the value of `kvikio::defaults::compat_mode()`. This overload only allows the
   * users to choose the compatibility mode from the `ON` (true) or `OFF` (false) option.
   *
   * Changing compatibility mode, effects all new FileHandles that doesn't sets the
   * `compat_mode` argument explicitly but it never effect existing FileHandles.
   *
   * @param enable Whether to enable compatibility mode or not.
   */
  static void compat_mode_reset(bool compat_mode)
  {
    if (compat_mode) {
      instance()->_compat_mode = CompatMode::ON;
    } else {
      instance()->_compat_mode = CompatMode::OFF;
    }
  }

  /**
   * @brief Reset the value of `kvikio::defaults::compat_mode()`. This overload allows the users to
   * choose the compatibility mode from one of the three options, including the `ALLOW` mode.
   *
   * Changing compatibility mode, effects all new FileHandles that doesn't sets the
   * `compat_mode` argument explicitly but it never effect existing FileHandles.
   *
   * @param enable Whether to enable compatibility mode or not.
   */
  static void compat_mode_reset(CompatMode compat_mode)
  {
    instance()->_compat_mode = compat_mode;
    if (compat_mode == CompatMode::ALLOW) { instance()->readjust_compat_mode(); }
  }

  /**
   * @brief Get the default thread pool.
   *
   * Notice, it is not possible to change the default thread pool. KvikIO will
   * always use the same thread pool however it is possible to change number of
   * threads in the pool (see `kvikio::default::thread_pool_nthreads_reset()`).
   *
   * @return The the default thread pool instance.
   */
  [[nodiscard]] static BS::thread_pool& thread_pool() { return instance()->_thread_pool; }

  /**
   * @brief Get the number of threads in the default thread pool.
   *
   * Set the default value using `kvikio::default::thread_pool_nthreads_reset()` or by
   * setting the `KVIKIO_NTHREADS` environment variable. If not set, the default value is 1.
   *
   * @return The number of threads.
   */
  [[nodiscard]] static unsigned int thread_pool_nthreads()
  {
    return thread_pool().get_thread_count();
  }

  /**
   * @brief Reset the number of threads in the default thread pool. Waits for all currently running
   * tasks to be completed, then destroys all threads in the pool and creates a new thread pool with
   * the new number of threads. Any tasks that were waiting in the queue before the pool was reset
   * will then be executed by the new threads. If the pool was paused before resetting it, the new
   * pool will be paused as well.
   *
   * @param nthreads The number of threads to use.
   */
  static void thread_pool_nthreads_reset(unsigned int nthreads)
  {
    if (nthreads == 0) {
      throw std::invalid_argument("number of threads must be a positive integer greater than zero");
    }
    thread_pool().reset(nthreads);
  }

  /**
   * @brief Get the default task size used for parallel IO operations.
   *
   * Set the default value using `kvikio::default::task_size_reset()` or by setting
   * the `KVIKIO_TASK_SIZE` environment variable. If not set, the default value is 4 MiB.
   *
   * @return The default task size in bytes.
   */
  [[nodiscard]] static std::size_t task_size() { return instance()->_task_size; }

  /**
   * @brief Reset the default task size used for parallel IO operations.
   *
   * @param nbytes The default task size in bytes.
   */
  static void task_size_reset(std::size_t nbytes)
  {
    if (nbytes == 0) {
      throw std::invalid_argument("task size must be a positive integer greater than zero");
    }
    instance()->_task_size = nbytes;
  }

  /**
   * @brief Get the default GDS threshold, which is the minimum size to use GDS (in bytes).
   *
   * In order to improve performance of small IO, `.pread()` and `.pwrite()` implement a shortcut
   * that circumvent the threadpool and use the POSIX backend directly.
   *
   * Set the default value using `kvikio::default::gds_threshold_reset()` or by setting the
   * `KVIKIO_GDS_THRESHOLD` environment variable. If not set, the default value is 1 MiB.
   *
   * @return The default GDS threshold size in bytes.
   */
  [[nodiscard]] static std::size_t gds_threshold() { return instance()->_gds_threshold; }

  /**
   * @brief Reset the default GDS threshold, which is the minimum size to use GDS (in bytes).
   * @param nbytes The default GDS threshold size in bytes.
   */
  static void gds_threshold_reset(std::size_t nbytes) { instance()->_gds_threshold = nbytes; }

  /**
   * @brief Get the size of the bounce buffer used to stage data in host memory.
   *
   * Set the value using `kvikio::default::bounce_buffer_size_reset()` or by setting the
   * `KVIKIO_BOUNCE_BUFFER_SIZE` environment variable. If not set, the value is 16 MiB.
   *
   * @return The bounce buffer size in bytes.
   */
  [[nodiscard]] static std::size_t bounce_buffer_size() { return instance()->_bounce_buffer_size; }

  /**
   * @brief Reset the size of the bounce buffer used to stage data in host memory.
   *
   * @param nbytes The bounce buffer size in bytes.
   */
  static void bounce_buffer_size_reset(std::size_t nbytes)
  {
    if (nbytes == 0) {
      throw std::invalid_argument(
        "size of the bounce buffer must be a positive integer greater than zero");
    }
    instance()->_bounce_buffer_size = nbytes;
  }
};

}  // namespace kvikio
