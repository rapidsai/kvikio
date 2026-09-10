/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/http_retry.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

HttpRetryPolicy::HttpRetryPolicy()
  : HttpRetryPolicy{defaults::http_max_attempts(), defaults::http_status_codes()}
{
}

HttpRetryPolicy::HttpRetryPolicy(std::size_t max_attempts, std::vector<int> retryable_status_codes)
  : _max_attempts{max_attempts}, _retryable_status_codes{std::move(retryable_status_codes)}
{
  KVIKIO_EXPECT(
    _max_attempts > 0, "max_attempts must be a positive integer", std::invalid_argument);
}

std::size_t HttpRetryPolicy::max_attempts() const noexcept { return _max_attempts; }

bool HttpRetryPolicy::is_retryable(CURLcode curl_code, long http_code) const
{
  // TODO: Currently the timeout is the only libcurl transport error treated as retryable. Need to
  // revisit and add more candidates.
  if (curl_code == CURLE_OPERATION_TIMEDOUT) { return true; }
  return std::find(_retryable_status_codes.begin(),
                   _retryable_status_codes.end(),
                   static_cast<int>(http_code)) != _retryable_status_codes.end();
}

std::chrono::milliseconds HttpRetryPolicy::backoff_for(std::size_t attempt) const
{
  // With a base value of 500ms, we retry after 500ms, 1s, 2s, 4s, ... (stays at 4s).
  auto const shift = std::min<std::size_t>(attempt - 1, 4);
  return std::min(_max_delay_ms, _base_delay_ms * (1 << shift));
}

RetryOutcome HttpRetryPolicy::evaluate(CURLcode curl_code,
                                       long http_code,
                                       std::size_t attempt,
                                       std::string_view curl_error_message,
                                       std::string_view error_prefix) const
{
  KVIKIO_EXPECT(attempt > 0, "attempt must be a positive integer", std::invalid_argument);

  // We set CURLE_HTTP_RETURNED_ERROR, so >= 400 status codes are considered errors, so anything
  // less than this is considered a success and we're done.
  if (curl_code == CURLE_OK) { return {RetryDecision::SUCCESS, std::chrono::milliseconds{0}, {}}; }

  if (!is_retryable(curl_code, http_code)) {
    std::stringstream ss;
    ss << error_prefix << " ";
    if (curl_error_message.empty()) {
      ss << "(" << curl_easy_strerror(curl_code) << ")";
    } else {
      ss << "(" << curl_error_message << ")";
    }
    return {RetryDecision::FATAL, std::chrono::milliseconds{0}, ss.str()};
  }

  if (attempt >= _max_attempts) {
    std::stringstream ss;
    ss << "KvikIO: HTTP request reached maximum number of attempts (" << _max_attempts
       << "). Reason: ";
    if (curl_code == CURLE_OPERATION_TIMEDOUT) {
      ss << "Operation timed out.";
    } else {
      ss << "Got HTTP code " << http_code << ".";
    }
    return {RetryDecision::EXHAUSTED, std::chrono::milliseconds{0}, ss.str()};
  }

  auto const delay_ms = backoff_for(attempt);
  std::stringstream ss;
  if (curl_code == CURLE_OPERATION_TIMEDOUT) {
    ss << "KvikIO: Timeout error. Retrying after " << delay_ms.count() << "ms (attempt " << attempt
       << " of " << _max_attempts << ").";
  } else {
    ss << "KvikIO: Got HTTP code " << http_code << ". Retrying after " << delay_ms.count()
       << "ms (attempt " << attempt << " of " << _max_attempts << ").";
  }
  return {RetryDecision::RETRY, delay_ms, ss.str()};
}

}  // namespace kvikio::detail
