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
#include <functional>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <kvikio/compat_mode.hpp>
#include <kvikio/error.hpp>
#include <kvikio/http_status_codes.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/threadpool_wrapper.hpp>

/**
 * @brief KvikIO namespace.
 */
namespace kvikio {

template <typename T>
std::optional<T> from_string(std::string const& env_val)
{
  std::stringstream ss(env_val);
  T converted_val;
  ss >> converted_val;

  if (!ss.fail()) { return converted_val; }

  // An exception: for string, empty value is allowed
  if constexpr (std::is_same_v<T, std::string>) { return std::optional<std::string>{""}; }

  // For all other cases, return std::nullopt
  return {};
}

template <typename T>
[[nodiscard]] T getenv_or(std::string_view env_var_name,
                          T default_val,
                          std::function<std::optional<T>(std::string const&)> conversion_callback,
                          std::map<T, std::vector<std::string>> const& dictionary,
                          bool case_sensitive,
                          std::function<std::optional<T>(std::string const&)> extra_callback)
{
  // Step 0: If the name does not exist, use default value
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  // Step 1: try to convert to type T
  std::optional<T> converted_val;
  if (conversion_callback) {
    converted_val = std::invoke(conversion_callback, env_val);
    if (converted_val.has_value()) { return converted_val.value(); }
  }

  // Step 2: look up in the user-provided dictionary
  std::string str{env_val};
  if (!dictionary.empty()) {
    // Convert to lowercase
    if (!case_sensitive) {
      // Special considerations regarding the case conversion:
      // - std::tolower() is not an addressable function. Passing it to std::transform() as
      //   a function pointer, if the compile turns out successful, causes the program behavior
      //   "unspecified (possibly ill-formed)", hence the lambda. ::tolower() is addressable
      //   and does not have this problem, but the following item still applies.
      // - To avoid UB in std::tolower() or ::tolower(), the character must be cast to unsigned
      // char.
      std::transform(
        str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    // Trim whitespaces
    std::stringstream trimmer;
    trimmer << str;
    str.clear();
    trimmer >> str;

    // Convert the dictionary to an easier format
    // Example:
    // dictionary is (v_1, {a, b}), (v_2, {d})
    // then flat_dictionary is (a, v_1), (b, v_1), (d, v_2)
    // and there must be no duplicate among a, b, d
    std::map<std::string, T> flat_dictionary;
    for (auto const& [dst, src_list] : dictionary) {
      for (auto const& src : src_list) {
        if (auto const it = flat_dictionary.find(src); it == flat_dictionary.end()) {
          flat_dictionary[src] = dst;
        } else {
          KVIKIO_FAIL("Duplicate environment variable values.");
        }
      }
    }

    // Look up in the dictionary
    if (auto it = flat_dictionary.find(str); it != flat_dictionary.end()) {
      return flat_dictionary[str];
    }
  }

  // Step 3: use extra user-provided callback
  if (extra_callback) {
    auto const result = std::invoke(extra_callback, str);
    return result.value();
  }

  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" + str,
              std::invalid_argument);
  return {};
}

template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  return getenv_or(env_var_name, default_val, from_string<T>, {}, true, {});
}

template <>
bool getenv_or(std::string_view env_var_name, bool default_val);

template <>
CompatMode getenv_or(std::string_view env_var_name, CompatMode default_val);

template <>
std::vector<int> getenv_or(std::string_view env_var_name, std::vector<int> default_val);

/**
 * @brief Get the environment variable value from a candidate list
 *
 * @tparam T Type of the environment variable value
 * @param env_var_names Candidate list containing the names of environment variable
 * @param default_val Default value of the environment variable, if none of the candidates has been
 * found
 * @return A tuple of (`env_var_name`, `result`, `has_found`), where:
 *   - If the environment variable is not set by any of the candidates, `has_found` will be false,
 * `result` will be `default_val`, and `env_var_name` will be empty.
 *   - If the environment variable is set by `env_var_name`, then `has_found` will be true, and
 * `result` be the set value. If more than one candidates have been set with the same value,
 * `env_var_name` will be assigned the last candidate.
 *
 * @throws std::invalid_argument if:
 *   - `env_var_names` is empty.
 *   - The environment variable is not defined to be string type and is assigned an empty value (in
 *     other words, string-type environment variables are allowed to hold an empty value).
 *   - More than one candidates have been set with different values.
 *   - An invalid value is given, e.g. value that cannot be converted to type T.
 */
template <typename T>
std::tuple<std::string_view, T, bool> getenv_or(
  std::initializer_list<std::string_view> env_var_names,
  T default_val,
  std::function<std::optional<T>(std::string const&)> conversion_callback = from_string<T>,
  std::map<T, std::vector<std::string>> dictionary                        = {},
  bool case_sensitive                                                     = false,
  std::function<std::optional<T>(std::string const&)> callback            = {})
{
  KVIKIO_EXPECT(env_var_names.size() > 0,
                "`env_var_names` must contain at least one environment variable name.",
                std::invalid_argument);
  std::string_view env_name_target;
  std::string_view env_val_str_target;
  T env_val_target;

  for (auto const& current_env_var_name : env_var_names) {
    auto const* current_env_val_str = std::getenv(current_env_var_name.data());
    if (current_env_val_str == nullptr) { continue; }

    auto current_env_val = getenv_or<T>(
      env_name_target, default_val, conversion_callback, dictionary, case_sensitive, callback);

    if (!env_name_target.empty() && env_val_target != current_env_val) {
      std::stringstream ss;
      ss << "Environment variable " << current_env_var_name << " (" << current_env_val_str
         << ") has already been set by its alias " << env_name_target << " (" << env_val_str_target
         << ") with a different value.";
      KVIKIO_FAIL(ss.str(), std::invalid_argument);
    }

    env_name_target    = current_env_var_name;
    env_val_target     = current_env_val;
    env_val_str_target = current_env_val_str;
  }

  if (env_name_target.empty()) { return {env_name_target, default_val, false}; }

  return {env_name_target, env_val_target, true};
}

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
  std::size_t _mmap_task_size;

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
   * @brief Alias of `thread_pool_nthreads`
   *
   * @return The number of threads
   */
  [[nodiscard]] static unsigned int num_threads();

  /**
   * @brief Alias of `set_thread_pool_nthreads`
   *
   * @param nthreads The number of threads to use
   */
  static void set_num_threads(unsigned int nthreads);

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

  [[nodiscard]] static std::size_t mmap_task_size();

  static void set_mmap_task_size(std::size_t nbytes);
};

}  // namespace kvikio
