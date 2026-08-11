/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/http_retry.hpp>
#include <kvikio/detail/parallel_operation.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/detail/tls.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

LibCurl::LibCurl()
{
  CURLcode err = curl_global_init(CURL_GLOBAL_DEFAULT);
  KVIKIO_EXPECT(err == CURLE_OK,
                "cannot initialize libcurl - errorcode: " + std::to_string(err),
                std::runtime_error);
  curl_version_info_data* ver = curl_version_info(::CURLVERSION_NOW);
  KVIKIO_EXPECT((ver->features & CURL_VERSION_THREADSAFE) != 0,
                "cannot initialize libcurl - built with thread safety disabled",
                std::runtime_error);
}

LibCurl::~LibCurl() noexcept
{
  _free_curl_handles.clear();
  curl_global_cleanup();
}

LibCurl& LibCurl::instance()
{
  static LibCurl _instance;
  return _instance;
}

LibCurl::UniqueHandlePtr LibCurl::get_free_handle()
{
  UniqueHandlePtr ret;
  std::lock_guard const lock(_mutex);
  if (!_free_curl_handles.empty()) {
    ret = std::move(_free_curl_handles.back());
    _free_curl_handles.pop_back();
  }
  return ret;
}

LibCurl::UniqueHandlePtr LibCurl::get_handle()
{
  // Check if we have a free handle available.
  UniqueHandlePtr ret = get_free_handle();
  if (ret) {
    curl_easy_reset(ret.get());
  } else {
    // If not, we create a new handle.
    CURL* raw_handle = curl_easy_init();
    KVIKIO_EXPECT(
      raw_handle != nullptr, "libcurl: call to curl_easy_init() failed", std::runtime_error);
    ret = UniqueHandlePtr(raw_handle, curl_easy_cleanup);
  }
  return ret;
}

void LibCurl::retain_handle(UniqueHandlePtr handle)
{
  std::lock_guard const lock(_mutex);
  _free_curl_handles.push_back(std::move(handle));
}

CurlHandle::CurlHandle(LibCurl::UniqueHandlePtr handle,
                       std::string source_file,
                       std::string source_line)
  : _handle{std::move(handle)}
{
  // Need CURLOPT_NOSIGNAL to support threading, see
  // <https://curl.se/libcurl/c/CURLOPT_NOSIGNAL.html>
  setopt(CURLOPT_NOSIGNAL, 1L);

  // We always set CURLOPT_ERRORBUFFER to get better error messages.
  _errbuf[0] = 0;  // Set the error buffer as empty.
  setopt(CURLOPT_ERRORBUFFER, _errbuf);

  // Make curl_easy_perform() fail when receiving HTTP code errors.
  setopt(CURLOPT_FAILONERROR, 1L);

  // Make requests time out after `value` seconds.
  setopt(CURLOPT_TIMEOUT, kvikio::defaults::http_timeout());

  // Optionally enable verbose output if it's configured.
  auto const verbose = getenv_or("KVIKIO_REMOTE_VERBOSE", false);
  if (verbose) { setopt(CURLOPT_VERBOSE, 1L); }

  detail::set_up_ca_paths(*this);
}

CurlHandle::~CurlHandle() noexcept { LibCurl::instance().retain_handle(std::move(_handle)); }

CURL* CurlHandle::handle() noexcept { return _handle.get(); }

std::string CurlHandle::error_message() const
{
  // Safe to construct from `_errbuf`: it is initialized empty in the constructor and libcurl always
  // writes null-terminated strings into it.
  return std::string{_errbuf};
}

void CurlHandle::clear_error_message() noexcept { _errbuf[0] = 0; }

void CurlHandle::perform() { perform({}); }

void CurlHandle::perform(std::function<void()> const& on_retry)
{
  // Snapshot the retry settings, so every attempt of this transfer follows the same policy.
  detail::HttpRetryPolicy const policy;

  for (std::size_t attempt = 1;; ++attempt) {
    clear_error_message();
    auto const curl_code = curl_easy_perform(handle());

    long http_code = 0;
    // We had an error. Is it retryable?
    if (curl_code != CURLE_OK) { getinfo(CURLINFO_RESPONSE_CODE, &http_code); }

    auto const outcome =
      policy.evaluate(curl_code, http_code, attempt, error_message(), "curl_easy_perform() error");
    switch (outcome.decision) {
      case detail::RetryDecision::SUCCESS: return;
      case detail::RetryDecision::RETRY:
        KVIKIO_LOG_WARN(outcome.message);
        if (on_retry) { on_retry(); }
        std::this_thread::sleep_for(outcome.delay_ms);
        break;
      default: KVIKIO_FAIL(outcome.message, std::runtime_error);
    }
  }
}
}  // namespace kvikio
