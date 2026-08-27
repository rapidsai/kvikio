/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include <curl/curl.h>

#include <kvikio/detail/curl_share.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

CurlShare::CurlShare()
{
  // Force LibCurl global init before we create the share handle.
  std::ignore = LibCurl::instance();

  _share = curl_share_init();
  KVIKIO_EXPECT(_share != nullptr, "curl_share_init() failed", std::runtime_error);

  auto set_option = [this](CURLSHoption option, auto value) {
    auto const sc = curl_share_setopt(_share, option, value);
    KVIKIO_EXPECT(sc == CURLSHE_OK,
                  std::string("curl_share_setopt: ") + curl_share_strerror(sc),
                  std::runtime_error);
  };
  set_option(CURLSHOPT_LOCKFUNC, &CurlShare::lock_callback);
  set_option(CURLSHOPT_UNLOCKFUNC, &CurlShare::unlock_callback);
  set_option(CURLSHOPT_USERDATA, this);
  set_option(CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);

  // Only DNS is shared.
  // libcurl does not support sharing connections, cookies or HSTS state across concurrent threads.
  // TLS sessions are excluded here for a different reason. A resumption ticket is consumed by
  // whoever takes it, and libcurl caps every cache at 2 tickets per remote endpoint. Merging the N
  // per-worker caches into one would therefore drop the tickets available for a given endpoint
  // from 2N to 2.
}

CurlShare& CurlShare::instance()
{
  // Leaked on purpose.
  static CurlShare* inst = new CurlShare();
  return *inst;
}

void CurlShare::lock_callback(CURL* /*handle*/,
                              curl_lock_data data,
                              curl_lock_access /*access*/,
                              void* userptr)
{
  auto* share = static_cast<CurlShare*>(userptr);
  share->_mutexes[static_cast<std::size_t>(data)].lock();
}

void CurlShare::unlock_callback(CURL* /*handle*/, curl_lock_data data, void* userptr)
{
  auto* share = static_cast<CurlShare*>(userptr);
  share->_mutexes[static_cast<std::size_t>(data)].unlock();
}

}  // namespace kvikio::detail
