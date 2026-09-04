/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
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
  static std::size_t const num_dns_caches = []() {
    auto const result = getenv_or<std::size_t>("KVIKIO_REMOTE_NUM_DNS_CACHES", 16);
    return std::max<std::size_t>(result, 1);
  }();

  // Leaked on purpose.
  static std::vector<CurlShareHandle*> const share_handles = [&]() {
    std::vector<CurlShareHandle*> result;
    result.reserve(num_dns_caches);
    for (std::size_t i = 0; i < num_dns_caches; ++i) {
      result.push_back(new CurlShareHandle());
    }
    return result;
  }();

  // Threads take the caches round-robin.
  static std::atomic<std::size_t> counter{0};
  thread_local std::size_t const assigned_index = counter.fetch_add(1) % num_dns_caches;
  return *share_handles[assigned_index];
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
