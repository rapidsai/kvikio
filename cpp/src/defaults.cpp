/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <BS_thread_pool.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/http_status_codes.hpp>
#include <kvikio/remote_handle.hpp>
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
  auto const normalized = detail::normalize_env_value(env_val);
  if (normalized == "true" || normalized == "on" || normalized == "yes") { return true; }
  if (normalized == "false" || normalized == "off" || normalized == "no") { return false; }
  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" + std::string{env_val},
              std::invalid_argument);
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

template <>
RemoteIOBackend getenv_or(std::string_view env_var_name, RemoteIOBackend default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  auto const normalized = detail::normalize_env_value(env_val);
  if (normalized == "easy_threadpool") { return RemoteIOBackend::EASY_THREADPOOL; }
  if (normalized == "multi_poll") { return RemoteIOBackend::MULTI_POLL; }
  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" + std::string{env_val},
              std::invalid_argument);
}

template <>
RemoteReactorDispatch getenv_or(std::string_view env_var_name, RemoteReactorDispatch default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  auto const normalized = detail::normalize_env_value(env_val);
  if (normalized == "per_chunk") { return RemoteReactorDispatch::PER_CHUNK; }
  if (normalized == "per_pread") { return RemoteReactorDispatch::PER_PREAD; }
  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" + std::string{env_val},
              std::invalid_argument);
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
  // Determine the default value of `num_bounce_buffers_per_cache`. 0 means unlimited.
  {
    ssize_t const env = getenv_or("KVIKIO_NUM_BOUNCE_BUFFERS_PER_CACHE", 16);
    KVIKIO_EXPECT(
      env >= 0,
      "KVIKIO_NUM_BOUNCE_BUFFERS_PER_CACHE has to be a non-negative integer (0 means unlimited)",
      std::invalid_argument);
    _num_bounce_buffers_per_cache = env;
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

  // Determine the default value of `auto_direct_io_read` and `auto_direct_io_write`
  {
    _auto_direct_io_read          = getenv_or("KVIKIO_AUTO_DIRECT_IO_READ", false);
    _auto_direct_io_read_overread = getenv_or("KVIKIO_AUTO_DIRECT_IO_READ_OVERREAD", false);
    _auto_direct_io_write         = getenv_or("KVIKIO_AUTO_DIRECT_IO_WRITE", true);
  }

  // Determine the default value of `thread_pool_per_block_device`
  {
    _thread_pool_per_block_device = getenv_or("KVIKIO_THREAD_POOL_PER_BLOCK_DEVICE", false);
  }

  // Determine the remote-IO backend selectors.
  {
    _remote_io_backend = getenv_or("KVIKIO_REMOTE_IO_BACKEND", RemoteIOBackend::EASY_THREADPOOL);
  }
  {
    ssize_t const env = getenv_or("KVIKIO_REMOTE_IO_NUM_REACTORS", 1);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_REMOTE_IO_NUM_REACTORS has to be a positive integer", std::invalid_argument);
    _remote_io_num_reactors = static_cast<unsigned int>(env);
  }
  {
    _remote_io_reactor_dispatch =
      getenv_or("KVIKIO_REMOTE_IO_REACTOR_DISPATCH", RemoteReactorDispatch::PER_CHUNK);
  }
  {
    ssize_t const env = getenv_or("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS", ssize_t{256});
    KVIKIO_EXPECT(
      env >= 0,
      "KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS has to be a non-negative integer (0 = unlimited)",
      std::invalid_argument);
    _remote_io_max_concurrent_requests = static_cast<std::size_t>(env);
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

ThreadPool& defaults::thread_pool() { return instance()->_thread_pool; }

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

std::size_t defaults::num_bounce_buffers_per_cache()
{
  return instance()->_num_bounce_buffers_per_cache;
}

void defaults::set_num_bounce_buffers_per_cache(std::size_t n)
{
  instance()->_num_bounce_buffers_per_cache = n;
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

bool defaults::auto_direct_io_read() { return instance()->_auto_direct_io_read; }

void defaults::set_auto_direct_io_read(bool flag) { instance()->_auto_direct_io_read = flag; }

bool defaults::auto_direct_io_read_overread() { return instance()->_auto_direct_io_read_overread; }

void defaults::set_auto_direct_io_read_overread(bool flag)
{
  instance()->_auto_direct_io_read_overread = flag;
}

bool defaults::auto_direct_io_write() { return instance()->_auto_direct_io_write; }

void defaults::set_auto_direct_io_write(bool flag) { instance()->_auto_direct_io_write = flag; }

bool defaults::thread_pool_per_block_device() { return instance()->_thread_pool_per_block_device; }

void defaults::set_thread_pool_per_block_device(bool flag)
{
  instance()->_thread_pool_per_block_device = flag;
}

RemoteIOBackend defaults::remote_io_backend() { return instance()->_remote_io_backend; }

unsigned int defaults::remote_io_num_reactors() { return instance()->_remote_io_num_reactors; }

RemoteReactorDispatch defaults::remote_io_reactor_dispatch()
{
  return instance()->_remote_io_reactor_dispatch;
}

std::size_t defaults::remote_io_max_concurrent_requests()
{
  return instance()->_remote_io_max_concurrent_requests;
}
}  // namespace kvikio
