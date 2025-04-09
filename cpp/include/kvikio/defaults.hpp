/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include <kvikio/compat_mode.hpp>
#include <kvikio/http_status_codes.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/threadpool_wrapper.hpp>

/**
 * @brief KvikIO namespace.
 */
namespace kvikio {

template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  KVIKIO_EXPECT(!sstream.fail(),
                "unknown config value " + std::string{env_var_name} + "=" + std::string{env_val},
                std::invalid_argument);
  return converted_val;
}

template <>
bool getenv_or(std::string_view env_var_name, bool default_val);

template <>
CompatMode getenv_or(std::string_view env_var_name, CompatMode default_val);

template <>
std::vector<int> getenv_or(std::string_view env_var_name, std::vector<int> default_val);

/**
 * @brief Singleton class of default values used throughout KvikIO.
 *
 */
class defaults {
 private:
  BS_thread_pool _thread_pool{get_num_threads_from_env()};
  CompatMode _compat_mode;
  std::size_t _task_size;
  std::size_t _gds_threshold;
  std::size_t _bounce_buffer_size;
  std::size_t _http_max_attempts;
  long _http_timeout;
  std::vector<int> _http_status_codes;

  static unsigned int get_num_threads_from_env();

  defaults();

  KVIKIO_EXPORT static defaults* instance();

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
   * @return Compatibility mode.
   */
  [[nodiscard]] static CompatMode compat_mode();

  /**
   * @brief Set the value of `kvikio::defaults::compat_mode()`.
   *
   * Changing the compatibility mode affects all the new FileHandles whose `compat_mode` argument is
   * not explicitly set, but it never affects existing FileHandles.
   *
   * @param compat_mode Compatibility mode.
   */
  static void set_compat_mode(CompatMode compat_mode);

  /**
   * @brief Infer the `AUTO` compatibility mode from the system runtime.
   *
   * If the requested compatibility mode is `AUTO`, set the expected compatibility mode to
   * `ON` or `OFF` by performing a system config check; otherwise, do nothing. Effectively, this
   * function reduces the requested compatibility mode from three possible states
   * (`ON`/`OFF`/`AUTO`) to two (`ON`/`OFF`) so as to determine the actual I/O path. This function
   * is lightweight as the inferred result is cached.
   */
  static CompatMode infer_compat_mode_if_auto(CompatMode compat_mode) noexcept;

  /**
   * @brief Given a requested compatibility mode, whether it is expected to reduce to `ON`.
   *
   * This function returns true if any of the two condition is satisfied:
   *   - The compatibility mode is `ON`.
   *   - It is `AUTO` but inferred to be `ON`.
   *
   * Conceptually, the opposite of this function is whether requested compatibility mode is expected
   * to be `OFF`, which would occur if any of the two condition is satisfied:
   *   - The compatibility mode is `OFF`.
   *   - It is `AUTO` but inferred to be `OFF`.
   *
   * @param compat_mode Compatibility mode.
   * @return Boolean answer.
   */
  static bool is_compat_mode_preferred(CompatMode compat_mode) noexcept;

  /**
   * @brief Whether the global compatibility mode from class defaults is expected to be `ON`.
   *
   * This function returns true if any of the two condition is satisfied:
   *   - The compatibility mode is `ON`.
   *   - It is `AUTO` but inferred to be `ON`.
   *
   * Conceptually, the opposite of this function is whether the global compatibility mode is
   * expected to be `OFF`, which would occur if any of the two condition is satisfied:
   *   - The compatibility mode is `OFF`.
   *   - It is `AUTO` but inferred to be `OFF`.
   *
   * @return Boolean answer.
   */
  static bool is_compat_mode_preferred();

  /**
   * @brief Get the default thread pool.
   *
   * Notice, it is not possible to change the default thread pool. KvikIO will
   * always use the same thread pool however it is possible to change number of
   * threads in the pool (see `kvikio::default::set_thread_pool_nthreads()`).
   *
   * @return The default thread pool instance.
   */
  [[nodiscard]] static BS_thread_pool& thread_pool();

  /**
   * @brief Get the number of threads in the default thread pool.
   *
   * Set the default value using `kvikio::default::set_thread_pool_nthreads()` or by
   * setting the `KVIKIO_NTHREADS` environment variable. If not set, the default value is 1.
   *
   * @return The number of threads.
   */
  [[nodiscard]] static unsigned int thread_pool_nthreads();

  /**
   * @brief Set the number of threads in the default thread pool. Waits for all currently running
   * tasks to be completed, then destroys all threads in the pool and creates a new thread pool with
   * the new number of threads. Any tasks that were waiting in the queue before the pool was reset
   * will then be executed by the new threads.
   *
   * @param nthreads The number of threads to use.
   */
  static void set_thread_pool_nthreads(unsigned int nthreads);

  /**
   * @brief Get the default task size used for parallel IO operations.
   *
   * Set the default value using `kvikio::default::set_task_size()` or by setting
   * the `KVIKIO_TASK_SIZE` environment variable. If not set, the default value is 4 MiB.
   *
   * @return The default task size in bytes.
   */
  [[nodiscard]] static std::size_t task_size();

  /**
   * @brief Set the default task size used for parallel IO operations.
   *
   * @param nbytes The default task size in bytes.
   */
  static void set_task_size(std::size_t nbytes);

  /**
   * @brief Get the default GDS threshold, which is the minimum size to use GDS (in bytes).
   *
   * In order to improve performance of small IO, `.pread()` and `.pwrite()` implement a shortcut
   * that circumvent the threadpool and use the POSIX backend directly.
   *
   * Set the default value using `kvikio::default::set_gds_threshold()` or by setting the
   * `KVIKIO_GDS_THRESHOLD` environment variable. If not set, the default value is 1 MiB.
   *
   * @return The default GDS threshold size in bytes.
   */
  [[nodiscard]] static std::size_t gds_threshold();

  /**
   * @brief Set the default GDS threshold, which is the minimum size to use GDS (in bytes).
   * @param nbytes The default GDS threshold size in bytes.
   */
  static void set_gds_threshold(std::size_t nbytes);

  /**
   * @brief Get the size of the bounce buffer used to stage data in host memory.
   *
   * Set the value using `kvikio::default::set_bounce_buffer_size()` or by setting the
   * `KVIKIO_BOUNCE_BUFFER_SIZE` environment variable. If not set, the value is 16 MiB.
   *
   * @return The bounce buffer size in bytes.
   */
  [[nodiscard]] static std::size_t bounce_buffer_size();

  /**
   * @brief Set the size of the bounce buffer used to stage data in host memory.
   *
   * @param nbytes The bounce buffer size in bytes.
   */
  static void set_bounce_buffer_size(std::size_t nbytes);

  /**
   * @brief Get the maximum number of attempts per remote IO read.
   *
   * Set the value using `kvikio::default::set_http_max_attempts()` or by setting
   * the `KVIKIO_HTTP_MAX_ATTEMPTS` environment variable. If not set, the value is 3.
   *
   * @return The maximum number of remote IO reads to attempt before raising an
   * error.
   */
  [[nodiscard]] static std::size_t http_max_attempts();

  /**
   * @brief Set the maximum number of attempts per remote IO read.
   *
   * @param attempts The maximum number of attempts to try before raising an error.
   */
  static void set_http_max_attempts(std::size_t attempts);

  /**
   * @brief The maximum time, in seconds, the transfer is allowed to complete.
   *
   * Set the value using `kvikio::default::set_http_timeout()` or by setting the
   * `KVIKIO_HTTP_TIMEOUT` environment variable. If not set, the value is 60.
   *
   * @return The maximum time the transfer is allowed to complete.
   */
  [[nodiscard]] static long http_timeout();

  /**
   * @brief Reset the http timeout.
   *
   * @param timeout_seconds The maximum time the transfer is allowed to complete.
   */
  static void set_http_timeout(long timeout_seconds);

  /**
   * @brief The list of HTTP status codes to retry.
   *
   * Set the value using `kvikio::default::set_http_status_codes()` or by setting the
   * `KVIKIO_HTTP_STATUS_CODES` environment variable. If not set, the default value is
   *
   * - 429
   * - 500
   * - 502
   * - 503
   * - 504
   *
   * @return The list of HTTP status codes to retry.
   */
  [[nodiscard]] static std::vector<int> const& http_status_codes();

  /**
   * @brief Set the list of HTTP status codes to retry.
   *
   * @param status_codes The HTTP status codes to retry.
   */
  static void set_http_status_codes(std::vector<int> status_codes);
};

}  // namespace kvikio
