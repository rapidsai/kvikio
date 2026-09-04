/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <mutex>

#include <curl/curl.h>

namespace kvikio::detail {

/**
 * @brief A libcurl share handle holding a DNS cache for a group of curl easy handles.
 *
 * An attached share handle overrides the multi handle's cache. `curl_multi_add_handle()` installs
 * its own only when the easy handle has none, so `CURLOPT_SHARE` has to be set first. Setting it in
 * the `CurlHandle` constructor guarantees that order.
 *
 * Setting the environment variable `KVIKIO_REMOTE_SHARE_DNS_CACHE` to a false value to disable
 * cache sharing and prevent this class from construction.
 */
class CurlShareHandle {
 public:
  /**
   * @brief Get the share handle serving the calling thread, creating the share handles on first
   * use.
   */
  static CurlShareHandle& share_handle_for_current_thread();

  CurlShareHandle(CurlShareHandle const&)            = delete;
  CurlShareHandle& operator=(CurlShareHandle const&) = delete;
  CurlShareHandle(CurlShareHandle&&)                 = delete;
  CurlShareHandle& operator=(CurlShareHandle&&)      = delete;

  /**
   * @brief The underlying libcurl share handle, for `curl_easy_setopt(CURLOPT_SHARE, ...)`.
   */
  [[nodiscard]] CURLSH* handle() const noexcept { return _share_handle; }

 private:
  CurlShareHandle();
  ~CurlShareHandle() = default;

  /**
   * @brief `CURLSHOPT_LOCKFUNC` callback. Takes the mutex guarding @p data.
   *
   * libcurl does no locking of its own, so without this callback the shared cache is accessed
   * unsynchronized.
   *
   * @param handle The easy handle libcurl is serving. Unused. The mutex is chosen by @p data alone.
   * @param data Which shared cache is about to be accessed.
   * @param access Whether libcurl wants shared or exclusive access. Unused, since the DNS cache is
   * locked exclusively.
   * @param userptr The `CurlShareHandle*` registered via `CURLSHOPT_USERDATA`.
   */
  static void lock_callback(CURL* handle,
                            curl_lock_data data,
                            curl_lock_access access,
                            void* userptr);

  /**
   * @brief `CURLSHOPT_UNLOCKFUNC` callback. Releases the mutex taken by `lock_callback`.
   *
   * @param handle The easy handle libcurl is serving. Unused.
   * @param data Which shared cache was accessed.
   * @param userptr The `CurlShareHandle*` registered via `CURLSHOPT_USERDATA`.
   */
  static void unlock_callback(CURL* handle, curl_lock_data data, void* userptr);

  CURLSH* _share_handle{nullptr};
  // One mutex per shareable data kind in libcurl
  std::array<std::mutex, CURL_LOCK_DATA_LAST> _mutexes;
};

}  // namespace kvikio::detail
