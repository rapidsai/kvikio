/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <curl/curl.h>

namespace kvikio::detail {

/**
 * @brief What the caller should do after one finished HTTP attempt.
 */
enum class RetryDecision {
  SUCCESS,   ///< The attempt succeeded. Nothing more to do.
  RETRY,     ///< Transfer failure with budget left. Wait and retry.
  FATAL,     ///< Non-retryable failure. Fail the transfer now.
  EXHAUSTED  ///< Retryable failure, but the attempt budget is spent.
};

/**
 * @brief The outcome on one finished HTTP attempt.
 */
struct RetryOutcome {
  RetryDecision decision{RetryDecision::FATAL};

  /**
   * @brief How long to wait before the next attempt. Meaningful only when `decision` is `RETRY`.
   */
  std::chrono::milliseconds delay_ms{0};

  /**
   * @brief The retry notice for `RETRY`, or the failure description for `FATAL` and `EXHAUSTED`.
   * Empty for `SUCCESS`.
   */
  std::string message;
};

/**
 * @brief Contains a snapshot of the HTTP retry settings, and the decision logic.
 */
class HttpRetryPolicy {
 public:
  /**
   * @brief Snapshot the HTTP retry settings `defaults::http_max_attempts()` and
   * `defaults::http_status_codes()`.
   */
  HttpRetryPolicy();

  /**
   * @brief Construct a policy with explicit settings, mainly for testing.
   *
   * @param max_attempts Maximum number of attempts, including the first. Must be positive.
   * @param retryable_status_codes HTTP status codes worth retrying.
   */
  HttpRetryPolicy(std::size_t max_attempts, std::vector<int> retryable_status_codes);

  /**
   * @brief Maximum number of attempts this policy allows, including the first.
   */
  [[nodiscard]] std::size_t max_attempts() const noexcept;

  /**
   * @brief Classify the outcome of one finished HTTP attempt.
   *
   * @param curl_code The libcurl result code for the attempt.
   * @param http_code `CURLINFO_RESPONSE_CODE`.
   * @param attempt 1-based index of the attempt that just finished. Must be positive.
   * @param curl_error_message The handle's `CURLOPT_ERRORBUFFER` text. Used only for the `FATAL`
   * description.
   * @param error_prefix Caller-specific prefix for the `FATAL` description.
   * @return The retry outcome.
   */
  [[nodiscard]] RetryOutcome evaluate(CURLcode curl_code,
                                      long http_code,
                                      std::size_t attempt,
                                      std::string_view curl_error_message,
                                      std::string_view error_prefix) const;

 private:
  /**
   * @brief Exponential backoff for the given 1-based attempt index.
   */
  [[nodiscard]] std::chrono::milliseconds backoff_for(std::size_t attempt) const;

  /**
   * @brief Whether a finished attempt is eligible for retry.
   */
  [[nodiscard]] bool is_retryable(CURLcode curl_code, long http_code) const;

  std::size_t _max_attempts;
  std::vector<int> _retryable_status_codes;

  std::chrono::milliseconds _base_delay_ms{500};
  std::chrono::milliseconds _max_delay_ms{4000};
};

}  // namespace kvikio::detail
