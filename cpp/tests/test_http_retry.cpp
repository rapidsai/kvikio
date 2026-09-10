/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/http_retry.hpp>

using kvikio::detail::HttpRetryPolicy;
using kvikio::detail::RetryDecision;

namespace {

constexpr std::string_view error_prefix{"curl_easy_perform() error"};

std::vector<int> default_status_codes() { return {429, 500, 502, 503, 504}; }

// A policy with a budget large enough that no attempt under test is ever EXHAUSTED, so the backoff
// schedule can be observed in isolation.
HttpRetryPolicy unlimited_policy() { return HttpRetryPolicy{1000, default_status_codes()}; }

}  // namespace

TEST(HttpRetryTest, retryable_curl_timeout)
{
  auto const outcome =
    unlimited_policy().evaluate(CURLE_OPERATION_TIMEDOUT, 0, 1, "", error_prefix);
  EXPECT_EQ(outcome.decision, RetryDecision::RETRY);
}

TEST(HttpRetryTest, retryable_http_code)
{
  auto const policy = unlimited_policy();
  for (auto const http_code : default_status_codes()) {
    auto const outcome = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, http_code, 1, "", error_prefix);
    EXPECT_EQ(outcome.decision, RetryDecision::RETRY);
  }
}

TEST(HttpRetryTest, non_retryable_failures)
{
  auto const policy = unlimited_policy();

  // Client errors outside the retryable list.
  for (auto const http_code : {400, 403, 404}) {
    auto const outcome = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, http_code, 1, "", error_prefix);
    EXPECT_EQ(outcome.decision, RetryDecision::FATAL);
  }

  // Transport errors outside the retryable set. A malformed URL and a failed certificate check
  // cannot succeed on a second attempt, and a write error is local to the caller.
  for (auto const curl_code :
       {CURLE_URL_MALFORMAT, CURLE_PEER_FAILED_VERIFICATION, CURLE_WRITE_ERROR}) {
    auto const outcome = policy.evaluate(curl_code, 0, 1, "", error_prefix);
    EXPECT_EQ(outcome.decision, RetryDecision::FATAL);
  }
}

TEST(HttpRetryTest, retryable_transport_failures)
{
  auto const policy = unlimited_policy();
  for (auto const curl_code : {CURLE_OPERATION_TIMEDOUT,
                               CURLE_COULDNT_RESOLVE_HOST,
                               CURLE_COULDNT_CONNECT,
                               CURLE_RECV_ERROR,
                               CURLE_SEND_ERROR,
                               CURLE_PARTIAL_FILE,
                               CURLE_GOT_NOTHING}) {
    auto const outcome = policy.evaluate(curl_code, 0, 1, "", error_prefix);
    EXPECT_EQ(outcome.decision, RetryDecision::RETRY) << "curl code " << curl_code;
  }
}

TEST(HttpRetryTest, success_needs_no_retry)
{
  auto const policy = unlimited_policy();
  // 200 answers the HEAD file-size probe, and also a server that ignores the range header.
  // 206 answers every range request the server honors, which is what a read gets.
  for (auto const http_code : {200, 206}) {
    auto const outcome = policy.evaluate(CURLE_OK, http_code, 1, "", error_prefix);
    EXPECT_EQ(outcome.decision, RetryDecision::SUCCESS);
    EXPECT_TRUE(outcome.message.empty());
    EXPECT_EQ(outcome.delay_ms, std::chrono::milliseconds{0});
  }
}

TEST(HttpRetryTest, backoff_doubles_then_saturates)
{
  // Attempts 5 and beyond exercise the clamp.
  std::vector<std::chrono::milliseconds> const expected{std::chrono::milliseconds{500},
                                                        std::chrono::milliseconds{1000},
                                                        std::chrono::milliseconds{2000},
                                                        std::chrono::milliseconds{4000},
                                                        std::chrono::milliseconds{4000},
                                                        std::chrono::milliseconds{4000}};

  auto const policy = unlimited_policy();
  for (std::size_t attempt = 1; attempt <= expected.size(); ++attempt) {
    auto const outcome = policy.evaluate(CURLE_OPERATION_TIMEDOUT, 0, attempt, "", error_prefix);
    ASSERT_EQ(outcome.decision, RetryDecision::RETRY) << "attempt " << attempt;
    EXPECT_EQ(outcome.delay_ms, expected[attempt - 1]) << "attempt " << attempt;
  }
}

TEST(HttpRetryTest, single_attempt_budget_never_retries)
{
  HttpRetryPolicy const policy{1, default_status_codes()};
  auto const outcome = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, 503, 1, "", error_prefix);
  EXPECT_EQ(outcome.decision, RetryDecision::EXHAUSTED);
}

TEST(HttpRetryTest, budget_is_spent_on_the_last_attempt)
{
  HttpRetryPolicy const policy{3, default_status_codes()};
  std::vector<RetryDecision> const expected{
    RetryDecision::RETRY, RetryDecision::RETRY, RetryDecision::EXHAUSTED};

  for (std::size_t attempt = 1; attempt <= expected.size(); ++attempt) {
    auto const outcome = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, 503, attempt, "", error_prefix);
    EXPECT_EQ(outcome.decision, expected[attempt - 1]) << "attempt " << attempt;
  }
}

TEST(HttpRetryTest, fatal_wins_over_a_remaining_budget)
{
  // A non-retryable failure stops the transfer even on the very first of many allowed attempts.
  HttpRetryPolicy const policy{10, default_status_codes()};
  auto const outcome = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, 404, 1, "", error_prefix);
  EXPECT_EQ(outcome.decision, RetryDecision::FATAL);
}

TEST(HttpRetryTest, retry_notice_text)
{
  HttpRetryPolicy const policy{3, default_status_codes()};

  auto const timed_out = policy.evaluate(CURLE_OPERATION_TIMEDOUT, 0, 1, "", error_prefix);
  EXPECT_EQ(timed_out.message,
            std::string{"KvikIO: Transport error: "} +
              curl_easy_strerror(CURLE_OPERATION_TIMEDOUT) +
              ". Retrying after 500ms (attempt 1 of 3).");

  auto const throttled = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, 503, 2, "", error_prefix);
  EXPECT_EQ(throttled.message,
            "KvikIO: Got HTTP code 503. Retrying after 1000ms (attempt 2 of 3).");
}

TEST(HttpRetryTest, exhausted_text)
{
  HttpRetryPolicy const policy{2, default_status_codes()};

  auto const timed_out = policy.evaluate(CURLE_OPERATION_TIMEDOUT, 0, 2, "", error_prefix);
  EXPECT_EQ(timed_out.message,
            std::string{"KvikIO: HTTP request reached maximum number of attempts (2). Reason: "
                        "Transport error: "} +
              curl_easy_strerror(CURLE_OPERATION_TIMEDOUT) + ".");

  auto const throttled = policy.evaluate(CURLE_HTTP_RETURNED_ERROR, 503, 2, "", error_prefix);
  EXPECT_EQ(throttled.message,
            "KvikIO: HTTP request reached maximum number of attempts (2). Reason: Got HTTP code "
            "503.");
}

TEST(HttpRetryTest, fatal_text)
{
  auto const policy = unlimited_policy();

  auto const with_errbuf = policy.evaluate(
    CURLE_HTTP_RETURNED_ERROR, 404, 1, "The requested URL returned error: 404", error_prefix);
  EXPECT_EQ(with_errbuf.message,
            "curl_easy_perform() error (The requested URL returned error: 404)");

  // Fall back to the generic description when libcurl recorded no message.
  auto const without_errbuf =
    policy.evaluate(CURLE_PEER_FAILED_VERIFICATION, 0, 1, "", error_prefix);
  EXPECT_EQ(without_errbuf.message,
            std::string{"curl_easy_perform() error ("} +
              curl_easy_strerror(CURLE_PEER_FAILED_VERIFICATION) + ")");
}

TEST(HttpRetryTest, invalid_arguments_are_rejected)
{
  EXPECT_THROW(HttpRetryPolicy(0, default_status_codes()), std::invalid_argument);

  auto const policy = unlimited_policy();
  EXPECT_THROW(std::ignore = policy.evaluate(CURLE_OPERATION_TIMEDOUT, 0, 0, "", error_prefix),
               std::invalid_argument);
}
