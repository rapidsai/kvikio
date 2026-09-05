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
struct Registry {
  struct Cache {
    CurlShareHandle* handle;
    std::size_t num_threads;  // How many live threads have been assigned to this cache.
  };
  std::mutex mutex;
  std::vector<Cache> caches;
};

struct Assignment {
  Registry* registry;
  std::size_t cache_idx;
  CurlShareHandle* handle;

  ~Assignment()
  {
    std::lock_guard const lock(registry->mutex);
    --registry->caches[cache_idx].num_threads;
  }
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
  static auto* const registry = new Registry{};

  thread_local Assignment const assignment = [&]() {
    std::lock_guard const lock(registry->mutex);
    for (std::size_t i = 0; i < registry->caches.size(); ++i) {
      if (registry->caches[i].num_threads < max_threads_per_cache) {
        ++registry->caches[i].num_threads;
        return Assignment{registry, i, registry->caches[i].handle};
      }
    }
    registry->caches.push_back({new CurlShareHandle(), 1});
    return Assignment{registry, registry->caches.size() - 1, registry->caches.back().handle};
  }();
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
