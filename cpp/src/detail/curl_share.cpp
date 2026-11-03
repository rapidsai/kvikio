/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/curl_share.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

namespace {

struct Assignment {
  CurlShareRegistry* registry;
  CurlShareHandle* handle;

  ~Assignment() { registry->release(handle); }
};
}  // namespace

CurlShareHandle::CurlShareHandle()
{
  // Force LibCurl global init before we create the share handle.
  std::ignore = LibCurl::instance();

  _share_handle = curl_share_init();
  KVIKIO_EXPECT(_share_handle != nullptr, "curl_share_init() failed", std::runtime_error);

  auto set_option = [this](CURLSHoption option, auto value) {
    auto const sc = curl_share_setopt(_share_handle, option, value);
    KVIKIO_EXPECT(sc == CURLSHE_OK,
                  std::string("curl_share_setopt: ") + curl_share_strerror(sc),
                  std::runtime_error);
  };
  set_option(CURLSHOPT_LOCKFUNC, &CurlShareHandle::lock_callback);
  set_option(CURLSHOPT_UNLOCKFUNC, &CurlShareHandle::unlock_callback);
  set_option(CURLSHOPT_USERDATA, this);
  set_option(CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);

  // Only DNS is shared.
  // libcurl does not support sharing connections, cookies or HSTS state across concurrent threads.
  // TLS sessions are excluded here for a different reason: A resumption ticket is consumed by
  // whoever takes it, and libcurl caps every cache at 2 tickets per remote endpoint. Merging the N
  // per-worker caches into one would therefore drop the tickets available for a given endpoint
  // from 2N to 2.
}

CurlShareRegistry::CurlShareRegistry(std::size_t max_threads_per_cache)
  : _max_threads_per_cache{max_threads_per_cache}
{
  KVIKIO_EXPECT(
    max_threads_per_cache > 0, "`max_threads_per_cache` must be positive", std::invalid_argument);
}

CurlShareHandle* CurlShareRegistry::acquire()
{
  std::lock_guard const lock(_mutex);
  for (auto& cache : _caches) {
    if (cache.num_threads < _max_threads_per_cache) {
      ++cache.num_threads;
      return cache.handle;
    }
  }
  // Leaked on purpose.
  _caches.push_back({new CurlShareHandle(), 1});
  return _caches.back().handle;
}

void CurlShareRegistry::release(CurlShareHandle* handle)
{
  std::lock_guard const lock(_mutex);
  for (auto& cache : _caches) {
    if (cache.handle == handle) {
      --cache.num_threads;
      return;
    }
  }
}

std::size_t CurlShareRegistry::num_caches() const
{
  std::lock_guard const lock(_mutex);
  return _caches.size();
}

CurlShareHandle& CurlShareHandle::share_handle_for_current_thread()
{
  static std::size_t const max_threads_per_cache = []() {
    ssize_t const env = getenv_or("KVIKIO_REMOTE_MAX_THREADS_PER_DNS_CACHE", ssize_t{16});
    KVIKIO_EXPECT(env >= 0,
                  "KVIKIO_REMOTE_MAX_THREADS_PER_DNS_CACHE has to be a non-negative integer",
                  std::invalid_argument);
    return std::max<std::size_t>(static_cast<std::size_t>(env), 1);
  }();

  // Leaked on purpose.
  static auto* const registry = new CurlShareRegistry(max_threads_per_cache);

  thread_local Assignment const assignment{registry, registry->acquire()};
  return *assignment.handle;
}

void CurlShareHandle::lock_callback(CURL* /*handle*/,
                                    curl_lock_data data,
                                    curl_lock_access /*access*/,
                                    void* userptr)
{
  auto* share_handle = static_cast<CurlShareHandle*>(userptr);
  share_handle->_mutexes[static_cast<std::size_t>(data)].lock();
}

void CurlShareHandle::unlock_callback(CURL* /*handle*/, curl_lock_data data, void* userptr)
{
  auto* share_handle = static_cast<CurlShareHandle*>(userptr);
  share_handle->_mutexes[static_cast<std::size_t>(data)].unlock();
}

}  // namespace kvikio::detail
