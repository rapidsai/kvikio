/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/observation.hpp>
#include <kvikio/shim/utils.hpp>

namespace KVIKIO_EXPORT kvikio {
namespace statistics {

/**
 * @brief Which rows `report()` prints.
 */
enum class ReportRows : std::uint8_t {
  /// Only the backends and subsystems the run used.
  USED,
  /// Every row, including the ones the run never used.
  ALL,
};

/**
 * @brief Totals for work that belongs to no single operation.
 *
 * An `Observation` records one operation, and part of what I/O costs does not fit there, either
 * because nothing correlates it with one operation or because it is shared between many.
 *
 * These counters count from the moment the process starts, whether or not anybody is watching, so
 * there is nothing to enable and nothing to switch off. Reading them is `counters()`, and the cost
 * of an interval is the difference between two readings.
 */
struct Counters {
  /// Remote file sizes asked for, which is an HTTP round trip each.
  std::uint64_t remote_size_probes{};
  /// Time spent waiting for them.
  Duration remote_size_probing{};

  /// Connections libcurl opened, as opposed to reused. Against the requests a run made, this is
  /// whether connections are being reused at all, which against a TLS endpoint dominates
  /// everything else.
  std::uint64_t http_connections{};
  /// Time those connections spent resolving, connecting, and shaking hands. libcurl measures
  /// these whether or not anybody asks, so reading them costs nothing.
  Duration http_dns{};
  Duration http_tcp{};
  Duration http_tls{};

  /// Requests the endpoint turned away with a retryable error, and the time spent sleeping before
  /// trying again. A request that was retried twice counts twice. Nothing else records this, since
  /// a retry that eventually succeeds is reported as a success.
  std::uint64_t http_retries{};
  Duration http_retry_backoff{};

  /**
   * @brief The difference between this reading and an earlier one.
   *
   * @param previous An earlier reading.
   * @return What was spent in between, saturating at zero.
   */
  [[nodiscard]] Counters since(Counters const& previous) const noexcept;

  /**
   * @brief Whether anything was counted at all.
   *
   * @return True if every counter is zero.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Serialise to JSON.
   *
   * @return A JSON object as a string.
   */
  [[nodiscard]] std::string to_json() const;

  /**
   * @brief Format a human-readable report.
   *
   * Grouped by subsystem, and a group the run never touched is left out, so an empty reading
   * formats as an empty string.
   *
   * @code
   *   http size probes     12 probes, 600 ms
   *   http handshake       128 connections, 40 ms dns, 900 ms tcp, 1.90 s tls
   * @endcode
   *
   * @param rows Which rows to print.
   * @return The report, newline-terminated, or empty.
   */
  [[nodiscard]] std::string report(ReportRows rows = ReportRows::USED) const;
};

/**
 * @brief Every counter as it stands now.
 *
 * @return The running totals.
 */
[[nodiscard]] Counters counters() noexcept;

}  // namespace statistics

namespace detail {

/**
 * @brief Times a scope and records the duration however the scope is left.
 *
 * @code
 * detail::ScopedTimer const probe{detail::count_remote_size_probe};
 * curl.perform();  // Counted whether it returns or throws.
 * @endcode
 *
 * @tparam Record What to hand the duration to. A `count_*()` below, or a lambda for one that
 * takes more than a duration.
 */
template <typename Record>
class ScopedTimer {
 public:
  explicit ScopedTimer(Record record) noexcept : _record{std::move(record)}, _started{now()} {}

  ~ScopedTimer() noexcept { _record(now() - _started); }

  ScopedTimer(ScopedTimer const&)            = delete;
  ScopedTimer& operator=(ScopedTimer const&) = delete;

 private:
  Record _record;
  TimePoint _started;
};

/**
 * @brief Record asking a remote endpoint how big a file is.
 *
 * @param probing How long the round trip took.
 */
void count_remote_size_probe(Duration probing) noexcept;

/**
 * @brief Record what a finished HTTP transfer spent getting connected.
 *
 * @param connections Connections opened, which is zero when one was reused.
 * @param dns Time resolving the name.
 * @param tcp Time establishing the transport connection.
 * @param tls Time shaking hands, or zero without TLS.
 */
void count_http_connection(std::uint64_t connections,
                           Duration dns,
                           Duration tcp,
                           Duration tls) noexcept;

/**
 * @brief Record an HTTP request that hit a retryable error and will be tried again.
 *
 * @param backoff How long the next attempt waits before it goes out.
 */
void count_http_retry(Duration backoff) noexcept;

}  // namespace detail
}  // namespace KVIKIO_EXPORT kvikio
