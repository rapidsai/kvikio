/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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

  // Receive buffer size, left at libcurl's CURL_MAX_WRITE_SIZE (16 KiB) default.
  //
  // Raising this looks obviously right and measures worse. At 16 KiB a 400G-class
  // NIC costs millions of recv() calls per second, and the syscall entry plus
  // hardened-usercopy check are visible in the profile. But libcurl allocates one
  // of these buffers per easy handle, and there is one handle per in-flight
  // sub-range, so the size multiplies by the concurrency: 4096 transfers hold
  // 64 MiB at 16 KiB and 4 GiB at 1 MiB. The smaller total stays inside L3 and
  // wins by more than the saved syscalls are worth. Measured on a g7e.48xlarge
  // across four cards at 4096-way concurrency: 862 Gbps at 16 KiB, 745 at
  // 256 KiB, 727 at 1 MiB.
  //
  // Exposed anyway because the trade-off inverts at low concurrency, where a few
  // fat transfers do prefer a large buffer. Zero leaves libcurl's default.
  //
  // These three options are read once rather than per handle, because a handle is
  // constructed per sub-range transfer and getenv() would then sit on the hot
  // path.
  static long const buffer_size = [] {
    auto const requested = getenv_or("KVIKIO_REMOTE_IO_BUFFER_SIZE", ssize_t{0});
    if (requested <= 0) { return 0L; }
    return static_cast<long>(std::clamp(requested, ssize_t{1024}, ssize_t{CURL_MAX_READ_SIZE}));
  }();
  if (buffer_size > 0) { setopt(CURLOPT_BUFFERSIZE, buffer_size); }

  // Bind every connection to one interface. The `if!<name>` form makes libcurl
  // use SO_BINDTODEVICE, which pins egress to a chosen NIC on a host with several
  // cards on one subnet. Without it the kernel picks by route metric and every
  // connection leaves through the same card.
  static std::string const interface_opt = [] {
    auto const* env = std::getenv("KVIKIO_REMOTE_IO_INTERFACE");
    return std::string{env == nullptr ? "" : env};
  }();
  if (!interface_opt.empty()) { setopt(CURLOPT_INTERFACE, interface_opt.c_str()); }

  // Spread connections over every address the resolver returns. S3 publishes many
  // front-end addresses, but libcurl connects to the first one that answers, so
  // without shuffling a whole process piles onto a single endpoint.
  static bool const shuffle_dns = getenv_or("KVIKIO_REMOTE_IO_DNS_SHUFFLE", false);
  if (shuffle_dns) { setopt(CURLOPT_DNS_SHUFFLE_ADDRESSES, 1L); }

  // Kernel TLS receive. Userspace TLS costs two touches of every byte: the
  // kernel copies ciphertext out to OpenSSL, then OpenSSL decrypts. At
  // multi-hundred-Gbps the copy alone is the largest single entry in the
  // profile. kTLS moves the record layer into the kernel so the payload is
  // decrypted on the way out to the caller.
  //
  // Off by default: it needs the `tls` module live, silently falls back to
  // userspace when the negotiated cipher is unsupported, and changes what is
  // being measured. Confirm it actually engaged by watching TlsCurrRxSw in
  // /proc/net/tls_stat rather than trusting the flag.
  static bool const enable_ktls = getenv_or("KVIKIO_REMOTE_IO_KTLS", false);
  if (enable_ktls) {
    setopt(CURLOPT_SSL_CTX_FUNCTION, +[](CURL*, void* ssl_ctx, void*) -> CURLcode {
      SSL_CTX_set_options(static_cast<SSL_CTX*>(ssl_ctx), SSL_OP_ENABLE_KTLS);
      return CURLE_OK;
    });
  }

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
