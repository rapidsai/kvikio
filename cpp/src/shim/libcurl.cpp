/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/parallel_operation.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/detail/tls.hpp>
#include <kvikio/error.hpp>
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

void CurlHandle::perform()
{
  long http_code          = 0;
  auto attempt_count      = 0;
  auto base_delay         = 500;   // milliseconds
  auto max_delay          = 4000;  // milliseconds
  auto http_max_attempts  = kvikio::defaults::http_max_attempts();
  auto& http_status_codes = kvikio::defaults::http_status_codes();
  CURLcode err;

  while (attempt_count++ < http_max_attempts) {
    err = curl_easy_perform(handle());

    if (err == CURLE_OK) {
      // We set CURLE_HTTP_RETURNED_ERROR, so >= 400 status codes are considered
      // errors, so anything less than this is considered a success and we're
      // done.
      return;
    }
    // We had an error. Is it retryable?
    curl_easy_getinfo(handle(), CURLINFO_RESPONSE_CODE, &http_code);
    auto const is_retryable_response =
      (std::find(http_status_codes.begin(), http_status_codes.end(), http_code) !=
       http_status_codes.end());

    if ((err == CURLE_OPERATION_TIMEDOUT) || is_retryable_response) {
      // backoff and retry again. With a base value of 500ms, we retry after
      // 500ms, 1s, 2s, 4s, ...
      auto const backoff_delay = base_delay * (1 << std::min(attempt_count - 1, 4));
      // up to a maximum of `max_delay` seconds.
      auto const delay = std::min(max_delay, backoff_delay);

      // Only print this message out and sleep if we're actually going to retry again.
      if (attempt_count < http_max_attempts) {
        if (err == CURLE_OPERATION_TIMEDOUT) {
          std::cout << "KvikIO: Timeout error. Retrying after " << delay << "ms (attempt "
                    << attempt_count << " of " << http_max_attempts << ")." << std::endl;
        } else {
          std::cout << "KvikIO: Got HTTP code " << http_code << ". Retrying after " << delay
                    << "ms (attempt " << attempt_count << " of " << http_max_attempts << ")."
                    << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      }
    } else {
      // We had some kind of fatal error, or we got some status code we don't retry.
      // We want to exit immediately.
      std::string msg(_errbuf);  // We can do this because we always initialize `_errbuf` as empty.
      std::stringstream ss;
      ss << "curl_easy_perform() error ";
      if (msg.empty()) {
        ss << "(" << curl_easy_strerror(err) << ")";
      } else {
        ss << "(" << msg << ")";
      }
      KVIKIO_FAIL(ss.str(), std::runtime_error);
    }
  }

  std::stringstream ss;
  ss << "KvikIO: HTTP request reached maximum number of attempts (" << http_max_attempts
     << "). Reason: ";
  if (err == CURLE_OPERATION_TIMEDOUT) {
    ss << "Operation timed out.";
  } else {
    ss << "Got HTTP code " << http_code << ".";
  }
  KVIKIO_FAIL(ss.str(), std::runtime_error);
}
}  // namespace kvikio
