/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <openssl/ssl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/curl_share.hpp>
#include <kvikio/detail/http_retry.hpp>
#include <kvikio/detail/parallel_operation.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/detail/tls.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/statistics/counters.hpp>
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

namespace {
/**
 * @brief Ask OpenSSL to offload TLS to the kernel. Called before the initialization of an SSL
 * connection, once all other SSL options are processed.
 *
 * @param ssl_ctx The `SSL_CTX` of the pending connection.
 * @return Always `CURLE_OK`. Requesting kTLS never fails. It is silently ignored when unavailable.
 */
CURLcode enable_ktls_callback(CURL*, void* ssl_ctx, void*)
{
  SSL_CTX_set_options(static_cast<SSL_CTX*>(ssl_ctx), SSL_OP_ENABLE_KTLS);
  return CURLE_OK;
}
}  // namespace

CurlHandle::CurlHandle(LibCurl::UniqueHandlePtr handle,
                       std::string source_file,
                       std::string source_line,
                       bool use_shared_dns_cache)
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

  // Resolve a hostname once per DNS cache.
  static bool const share_dns_cache = getenv_or("KVIKIO_REMOTE_SHARE_DNS_CACHE", true);
  if (use_shared_dns_cache && share_dns_cache) {
    setopt(CURLOPT_SHARE, detail::CurlShareHandle::share_handle_for_current_thread().handle());
  } else {
    setopt(CURLOPT_SHARE, static_cast<CURLSH*>(nullptr));
  }

  // Optionally enable verbose output if it's configured.
  static bool const verbose = getenv_or("KVIKIO_REMOTE_VERBOSE", false);
  if (verbose) { setopt(CURLOPT_VERBOSE, 1L); }

  // Bind every connection to one interface. Otherwise the kernel routes them all out the
  // lowest-metric NIC when several sit on one subnet. Passed to libcurl verbatim. A bare `<ip>`
  // binds the source address and leaves the egress NIC to policy routing, while `if!<name>`
  // binds the device with SO_BINDTODEVICE. A bad value fails at connection time with
  // CURLE_INTERFACE_FAILED, not here.
  static std::string const interface_opt = [] {
    auto const* env = std::getenv("KVIKIO_REMOTE_INTERFACE");
    return std::string{env == nullptr ? "" : env};
  }();
  if (!interface_opt.empty()) { setopt(CURLOPT_INTERFACE, interface_opt.c_str()); }

  // Shuffle the resolved addresses to spread connections over S3 front-ends. Addresses are not
  // reshuffled if name resolution is completed using the DNS cache. Therefore the spread is only as
  // wide as the number of caches (KVIKIO_REMOTE_NUM_DNS_CACHES).
  static bool const shuffle_dns = getenv_or("KVIKIO_REMOTE_DNS_SHUFFLE", false);
  if (shuffle_dns) { setopt(CURLOPT_DNS_SHUFFLE_ADDRESSES, 1L); }

  // How long resolved addresses stay cached, or -1 to keep them forever.
  static long const dns_cache_timeout = getenv_or<long>("KVIKIO_REMOTE_DNS_CACHE_TIMEOUT", 60);
  setopt(CURLOPT_DNS_CACHE_TIMEOUT, dns_cache_timeout);

  // Kernel TLS: decrypt in the kernel so the payload is touched once instead of twice (copied out
  // to OpenSSL, then decrypted). Enables both directions, though only receive matters here.
  // Off by default because it needs the `tls` kernel module and an OpenSSL built with
  // `enable-ktls`. It silently stays in userspace when either is missing, or when the negotiated
  // cipher is unsupported. Confirm it engaged via /proc/net/tls_stat, not this flag.
  static bool const enable_ktls = getenv_or("KVIKIO_REMOTE_KTLS", false);
  if (enable_ktls) { setopt(CURLOPT_SSL_CTX_FUNCTION, enable_ktls_callback); }

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

namespace detail {
namespace {
/// A libcurl timing in microseconds, or zero if it could not be read. libcurl fills the value in
/// only when the call succeeds, so a failed one leaves the phase uncounted rather than garbage.
[[nodiscard]] curl_off_t timing_of(CURL* easy, CURLINFO info) noexcept
{
  curl_off_t value{0};
  if (curl_easy_getinfo(easy, info, &value) != CURLE_OK) { return 0; }
  return value;
}
}  // namespace

void count_http_connection_of(CURL* easy) noexcept
{
  using std::chrono::microseconds;

  long connections{0};
  if (curl_easy_getinfo(easy, CURLINFO_NUM_CONNECTS, &connections) != CURLE_OK) { return; }
  // Zero means the connection was reused, so nothing was paid here.
  if (connections <= 0) { return; }

  auto const namelookup = timing_of(easy, CURLINFO_NAMELOOKUP_TIME_T);
  auto const connect    = timing_of(easy, CURLINFO_CONNECT_TIME_T);
  auto const appconnect = timing_of(easy, CURLINFO_APPCONNECT_TIME_T);

  auto const tcp = connect > namelookup ? microseconds{connect - namelookup} : Duration::zero();
  auto const tls = appconnect > connect ? microseconds{appconnect - connect} : Duration::zero();
  count_http_connection(
    static_cast<std::uint64_t>(connections), microseconds{namelookup}, tcp, tls);
}

}  // namespace detail

void CurlHandle::clear_error_message() noexcept { _errbuf[0] = 0; }

void CurlHandle::perform() { perform({}); }

void CurlHandle::perform(std::function<void()> const& on_retry)
{
  // Snapshot the retry settings, so every attempt of this transfer follows the same policy.
  detail::HttpRetryPolicy const policy;

  for (std::size_t attempt = 1;; ++attempt) {
    clear_error_message();
    auto const curl_code = curl_easy_perform(handle());
    detail::count_http_connection_of(handle());

    long http_code = 0;
    // We had an error. Is it retryable?
    if (curl_code != CURLE_OK) { getinfo(CURLINFO_RESPONSE_CODE, &http_code); }

    auto const outcome =
      policy.evaluate(curl_code, http_code, attempt, error_message(), "curl_easy_perform() error");
    switch (outcome.decision) {
      case detail::RetryDecision::SUCCESS: return;
      case detail::RetryDecision::RETRY:
        KVIKIO_LOG_WARN(outcome.message);
        detail::count_http_retry(outcome.delay_ms);
        if (on_retry) { on_retry(); }
        std::this_thread::sleep_for(outcome.delay_ms);
        break;
      default: KVIKIO_FAIL(outcome.message, std::runtime_error);
    }
  }
}
}  // namespace kvikio
