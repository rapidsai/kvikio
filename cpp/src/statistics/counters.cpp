/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/detail/string_utils.hpp>
#include <kvikio/statistics/counters.hpp>

namespace kvikio {

namespace {

/**
 * @brief The counters themselves.
 *
 * Relaxed throughout. Each is a running total that nothing else is ordered against, and a reading
 * taken while another thread is incrementing is a snapshot of a moving target either way.
 */
struct Atomics {
  std::atomic<std::uint64_t> remote_size_probes{0};
  std::atomic<std::int64_t> remote_size_probing_ns{0};
  std::atomic<std::uint64_t> http_connections{0};
  std::atomic<std::int64_t> http_dns_ns{0};
  std::atomic<std::int64_t> http_tcp_ns{0};
  std::atomic<std::int64_t> http_tls_ns{0};
  std::atomic<std::uint64_t> http_retries{0};
  std::atomic<std::int64_t> http_retry_backoff_ns{0};
};

/**
 * @brief The one set of counters.
 *
 * Intentionally leaked, since a thread pool can be torn down during static destruction and must
 * still find somewhere to count.
 */
Atomics& atomics() noexcept
{
  static auto* instance = new Atomics{};
  return *instance;
}

/// Add a duration to a counter of nanoseconds.
void add(std::atomic<std::int64_t>& counter, Duration duration) noexcept
{
  counter.fetch_add(duration.count(), std::memory_order_relaxed);
}

}  // namespace

namespace detail {

void count_remote_size_probe(Duration probing) noexcept
{
  auto& all = atomics();
  all.remote_size_probes.fetch_add(1, std::memory_order_relaxed);
  add(all.remote_size_probing_ns, probing);
}

void count_http_connection(std::uint64_t connections,
                           Duration dns,
                           Duration tcp,
                           Duration tls) noexcept
{
  auto& all = atomics();
  all.http_connections.fetch_add(connections, std::memory_order_relaxed);
  add(all.http_dns_ns, dns);
  add(all.http_tcp_ns, tcp);
  add(all.http_tls_ns, tls);
}

void count_http_retry(Duration backoff) noexcept
{
  auto& all = atomics();
  all.http_retries.fetch_add(1, std::memory_order_relaxed);
  add(all.http_retry_backoff_ns, backoff);
}

}  // namespace detail

namespace statistics {

Counters counters() noexcept
{
  auto& all = atomics();
  Counters ret;
  ret.remote_size_probes  = all.remote_size_probes.load(std::memory_order_relaxed);
  ret.remote_size_probing = Duration{all.remote_size_probing_ns.load(std::memory_order_relaxed)};
  ret.http_connections    = all.http_connections.load(std::memory_order_relaxed);
  ret.http_dns            = Duration{all.http_dns_ns.load(std::memory_order_relaxed)};
  ret.http_tcp            = Duration{all.http_tcp_ns.load(std::memory_order_relaxed)};
  ret.http_tls            = Duration{all.http_tls_ns.load(std::memory_order_relaxed)};
  ret.http_retries        = all.http_retries.load(std::memory_order_relaxed);
  ret.http_retry_backoff  = Duration{all.http_retry_backoff_ns.load(std::memory_order_relaxed)};
  return ret;
}

namespace {
// Saturating, since a counter only grows but the two readings need not be ordered.
template <typename T>
[[nodiscard]] constexpr T spent(T mine, T previous) noexcept
{
  return mine > previous ? mine - previous : T{};
}
}  // namespace

Counters Counters::since(Counters const& previous) const noexcept
{
  // Every counter is named below. Adding one to the struct changes its size, which fails here
  // until it is named too.
  static_assert(sizeof(Counters) == 8 * sizeof(std::uint64_t),
                "a counter was added or removed: difference it below and correct this size");
  return Counters{
    .remote_size_probes  = spent(remote_size_probes, previous.remote_size_probes),
    .remote_size_probing = spent(remote_size_probing, previous.remote_size_probing),
    .http_connections    = spent(http_connections, previous.http_connections),
    .http_dns            = spent(http_dns, previous.http_dns),
    .http_tcp            = spent(http_tcp, previous.http_tcp),
    .http_tls            = spent(http_tls, previous.http_tls),
    .http_retries        = spent(http_retries, previous.http_retries),
    .http_retry_backoff  = spent(http_retry_backoff, previous.http_retry_backoff),
  };
}

bool Counters::empty() const noexcept
{
  return remote_size_probes == 0 && http_connections == 0 && http_retries == 0;
}

std::string Counters::to_json() const
{
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << "{\"remote_size_probes\": " << remote_size_probes
     << ", \"remote_size_probing_ns\": " << remote_size_probing.count()
     << ", \"http_connections\": " << http_connections << ", \"http_dns_ns\": " << http_dns.count()
     << ", \"http_tcp_ns\": " << http_tcp.count() << ", \"http_tls_ns\": " << http_tls.count()
     << ", \"http_retries\": " << http_retries
     << ", \"http_retry_backoff_ns\": " << http_retry_backoff.count() << "}";
  return os.str();
}

std::string Counters::report(ReportRows rows) const
{
  std::ostringstream os;
  // The same width the summary's rows use, so the two read as one report.
  auto const row = [&os](char const* label, std::string const& value) {
    os << "  " << std::left << std::setw(21) << label << value << "\n";
  };

  // All of it or none of it. A zero here is an answer, `0 retries` or nothing spent on TLS, once
  // the run has done any remote I/O at all.
  if (rows != ReportRows::ALL && empty()) { return os.str(); }

  {
    std::ostringstream value;
    value << remote_size_probes << " probes, " << detail::format_duration(remote_size_probing);
    row("http size probes", value.str());
  }
  {
    std::ostringstream value;
    value << http_connections << " connections, " << detail::format_duration(http_dns) << " dns, "
          << detail::format_duration(http_tcp) << " tcp, " << detail::format_duration(http_tls)
          << " tls";
    row("http handshake", value.str());
  }
  {
    std::ostringstream value;
    value << http_retries << " retries, " << detail::format_duration(http_retry_backoff)
          << " backoff";
    row("http retries", value.str());
  }
  return os.str();
}

}  // namespace statistics
}  // namespace kvikio
