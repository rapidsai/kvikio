/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/mmap.hpp>
#include <kvikio/observation.hpp>
#include <kvikio/statistics/summary.hpp>

#include "utils/utils.hpp"

using kvikio::IoBackend;
using kvikio::MemoryKind;
using kvikio::Observation;
using kvikio::ObservationKind;
using kvikio::TransferDirection;

class SummaryTest : public testing::Test {
 protected:
  void SetUp() override
  {
    _filepath = _tmp_dir.path() / "test_observation";
    _data.resize(_num_elements);
    std::iota(_data.begin(), _data.end(), 0);

    kvikio::FileHandle f{_filepath, "w"};
    f.pwrite(_data.data(), nbytes()).get();
  }

  [[nodiscard]] std::size_t nbytes() const { return _data.size() * sizeof(std::uint64_t); }

  kvikio::test::TempDir _tmp_dir{};
  std::string _filepath;
  // 8 MiB, enough to be split into several tasks at the default task size.
  static constexpr std::size_t _num_elements = 1024ull * 1024ull;
  std::vector<std::uint64_t> _data;
};

using kvikio::statistics::ReportRows;
using kvikio::statistics::Summary;
using kvikio::statistics::SummaryMonitor;

TEST_F(SummaryTest, a_monitor_counts_the_calls_it_spans)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  EXPECT_EQ(monitor.get().num_ops, 0);
  {
    kvikio::FileHandle w{_filepath, "r+"};
    w.pwrite(buffer.data(), nbytes(), 0).get();
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
    f.pread(buffer.data(), nbytes(), 0).get();
  }

  auto const s = monitor.get();
  // Three calls, however many reads and writes they were split into.
  EXPECT_EQ(s.num_ops, 3);
  EXPECT_EQ(s.bytes_transferred, 3 * nbytes());
  EXPECT_EQ(s.bytes_requested, 3 * nbytes());
  EXPECT_EQ(s.num_errors, 0);
  EXPECT_EQ(s.num_reads, 2);
  EXPECT_EQ(s.num_writes, 1);
  EXPECT_EQ(s.num_reads + s.num_writes, s.num_ops);
  EXPECT_EQ(s.bytes_read, 2 * nbytes());
  EXPECT_EQ(s.bytes_written, nbytes());
  EXPECT_EQ(s.bytes_read + s.bytes_written, s.bytes_transferred);
  EXPECT_GT(s.wall(), kvikio::Duration::zero());
  EXPECT_GT(s.busy, kvikio::Duration::zero());
  EXPECT_GT(s.busy_bytes_per_sec(), 0.0);
  EXPECT_GT(s.busy_fraction(), 0.0);
  EXPECT_LE(s.busy_fraction(), 1.0);
  // Each call was waited on before the next was made, so nothing overlapped and the durations add
  // up to no more than the time anything was in flight.
  EXPECT_LE(s.total_duration, s.busy);
  EXPECT_GT(s.mean_duration(), kvikio::Duration::zero());
}

TEST_F(SummaryTest, a_span_that_did_remote_io_reports_the_http_rows_even_when_they_are_zero)
{
  SummaryMonitor const monitor;
  {
    kvikio::detail::LogicalObservationRecorder remote{
      IoBackend::REMOTE_HTTP, TransferDirection::READ, MemoryKind::HOST, 0, 1024};
    remote.finish(1024);
  }

  auto const s = monitor.get();
  // Nothing was probed, connected or retried inside the span, which is what perfect reuse of a
  // connection opened before it looks like, and is worth a row rather than a silence.
  EXPECT_TRUE(s.counters.empty());
  auto const text = s.report();
  EXPECT_NE(text.find("http handshake       0 connections"), std::string::npos);
  EXPECT_NE(text.find("http retries         0 retries"), std::string::npos);
  // A span that never left the machine still gets none of them.
  EXPECT_EQ(Summary{}.report().find("http"), std::string::npos);
}

TEST_F(SummaryTest, the_backends_that_carried_the_work_are_counted_apart)
{
  SummaryMonitor const monitor;
  {
    kvikio::detail::LogicalObservationRecorder gds{
      IoBackend::GDS, TransferDirection::READ, MemoryKind::DEVICE, 0, 1024};
    gds.finish(1024);
    kvikio::detail::LogicalObservationRecorder posix_a{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, 512};
    posix_a.finish(512);
    kvikio::detail::LogicalObservationRecorder posix_b{
      IoBackend::POSIX, TransferDirection::WRITE, MemoryKind::HOST, 0, 256};
    posix_b.finish(256);
  }

  auto const s      = monitor.get();
  auto const& gds   = s.by_backend[static_cast<std::size_t>(IoBackend::GDS)];
  auto const& posix = s.by_backend[static_cast<std::size_t>(IoBackend::POSIX)];
  auto const& mmap  = s.by_backend[static_cast<std::size_t>(IoBackend::MMAP)];
  EXPECT_EQ(gds.num_ops, 1);
  EXPECT_EQ(gds.bytes_transferred, 1024);
  EXPECT_EQ(posix.num_ops, 2);
  EXPECT_EQ(posix.bytes_transferred, 512 + 256);
  EXPECT_EQ(mmap.num_ops, 0) << "a backend that carried nothing is not counted";
  // Every operation belongs to exactly one backend, so the parts add up to the whole.
  EXPECT_EQ(gds.num_ops + posix.num_ops, s.num_ops);
  EXPECT_EQ(gds.bytes_transferred + posix.bytes_transferred, s.bytes_transferred);
  EXPECT_EQ(gds.total_duration + posix.total_duration, s.total_duration);
  // Compatibility mode decides this per call, so the report says it outright, one row each.
  auto const text = s.report();
  EXPECT_NE(text.find("backend GDS          1 KiB in 1 ops"), std::string::npos);
  EXPECT_NE(text.find("backend POSIX        768 B in 2 ops"), std::string::npos);
  // A backend the run never reached is left out, and says so when every row is asked for.
  EXPECT_EQ(text.find("backend MMAP"), std::string::npos);
  EXPECT_NE(s.report(ReportRows::ALL).find("backend MMAP         unused"), std::string::npos);
  EXPECT_NE(s.to_json().find("\"by_backend\""), std::string::npos);
}

namespace {

/// A monitor that stalls the first completion it is given, so that a later one overtakes it.
class Delayer final : public kvikio::Monitor {
 public:
  void on_start(Observation const&) noexcept override {}
  void on_finish(Observation const&) noexcept override
  {
    if (!_stalled.exchange(true)) { std::this_thread::sleep_for(std::chrono::milliseconds{60}); }
  }

 private:
  std::atomic<bool> _stalled{false};
};

}  // namespace

TEST_F(SummaryTest, completions_reported_out_of_order_are_still_counted_exactly)
{
  Delayer delayer;
  // Registered ahead of ours, so it is notified first and can hold a completion back.
  auto const id = kvikio::register_monitor(&delayer);

  SummaryMonitor const monitor;
  auto first = std::make_unique<kvikio::detail::LogicalObservationRecorder>(
    IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes());
  auto second = std::make_unique<kvikio::detail::LogicalObservationRecorder>(
    IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes());

  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  // Ends first and is held in the delayer, so the one that ends later arrives before it.
  std::thread early{[&] { first->finish(nbytes()); }};
  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  second->finish(nbytes());
  early.join();
  kvikio::unregister_monitor(id);

  auto const s = monitor.get();
  ASSERT_EQ(s.num_ops, 2);
  // The stretch closes at the latest end of the operations that were in it, whatever order the
  // reports arrived in, so busy covers the two of them and no more.
  EXPECT_LE(s.busy, s.wall());
  EXPECT_LT(s.busy, s.total_duration) << "they overlapped, so busy counts the overlap once";
}

TEST_F(SummaryTest, what_arrived_is_reported_against_what_was_asked_for)
{
  SummaryMonitor const monitor;
  {
    kvikio::detail::LogicalObservationRecorder whole{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, 1024};
    whole.finish(1024);
  }
  // The same shape whether or not anything came up short, so two runs can be compared line by
  // line rather than one of them growing a row the other lacks.
  EXPECT_NE(monitor.get().report().find("bytes                1 KiB of 1 KiB requested"),
            std::string::npos);

  {
    kvikio::detail::LogicalObservationRecorder partial{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, 1024};
    partial.finish(768);
  }
  auto const s = monitor.get();
  EXPECT_EQ(s.bytes_requested, 2048);
  EXPECT_EQ(s.bytes_transferred, 1024 + 768);
  EXPECT_NE(s.report().find("bytes                1.75 KiB of 2 KiB requested"), std::string::npos);
}

TEST_F(SummaryTest, a_monitor_counts_from_where_it_started_and_only_its_own)
{
  std::vector<std::uint64_t> buffer(_data.size());
  kvikio::FileHandle f{_filepath, "r"};
  f.pread(buffer.data(), nbytes(), 0).get();  // Before either monitor exists.

  SummaryMonitor a;
  f.pread(buffer.data(), nbytes(), 0).get();
  SummaryMonitor const b;
  f.pread(buffer.data(), nbytes(), 0).get();

  EXPECT_EQ(a.get().num_ops, 2) << "the read before the monitor was counted";
  EXPECT_EQ(b.get().num_ops, 1);
  EXPECT_EQ(a.get().bytes_transferred, 2 * nbytes());

  a.reset();
  EXPECT_EQ(a.get().num_ops, 0);
  EXPECT_EQ(b.get().num_ops, 1) << "resetting one must not touch the other";
}

TEST_F(SummaryTest, a_failed_call_is_counted_as_an_error)
{
  SummaryMonitor const monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    EXPECT_ANY_THROW(f.pwrite(_data.data(), nbytes(), 0).get());
  }
  auto const s = monitor.get();
  EXPECT_EQ(s.num_ops, 1);
  EXPECT_EQ(s.num_errors, 1);
  EXPECT_EQ(s.bytes_transferred, 0) << "a failed call moved nothing";
}

TEST_F(SummaryTest, busy_time_ignores_idle)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto const s = monitor.get();

  // The sleep grows the wall span but not the busy time, which is the whole point: the busy rate
  // describes the I/O while the whole-span rate describes the program.
  EXPECT_LT(s.busy, s.wall());
  EXPECT_LT(s.busy_fraction(), 0.9);
  EXPECT_GT(
    s.busy_bytes_per_sec(),
    static_cast<double>(s.bytes_transferred) / std::chrono::duration<double>{s.wall()}.count());
}

TEST_F(SummaryTest, a_snapshot_is_self_consistent)
{
  // Every field comes from one critical section, so a reader racing with completing calls can
  // never see busy time exceeding the span it is measured against.
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  kvikio::FileHandle f{_filepath, "r"};

  std::atomic<bool> stop{false};
  std::thread reader{[&monitor, &stop]() {
    while (!stop.load()) {
      auto const s = monitor.get();
      EXPECT_LE(s.busy, s.wall());
      EXPECT_LE(s.busy_fraction(), 1.0);
      EXPECT_LE(s.bytes_transferred, s.bytes_requested);
    }
  }};
  for (int i = 0; i < 20; ++i) {
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  stop = true;
  reader.join();
}

TEST_F(SummaryTest, consecutive_intervals_partition_the_run)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  kvikio::FileHandle f{_filepath, "r"};

  // What a periodic reporter does: one reading per tick, each differenced against the last. Taking
  // the reading and the interval separately would leave a gap between them, and an operation that
  // completed in that gap would land in both intervals.
  auto const first = monitor.get();
  auto baseline    = first;
  std::vector<Summary> intervals;
  for (int tick = 0; tick < 4; ++tick) {
    f.pread(buffer.data(), nbytes(), 0).get();
    auto const now = monitor.get();
    intervals.push_back(now.since(baseline));
    baseline = now;
  }

  // One interval describes its own tick, not the run so far.
  auto const& one = intervals.front();
  EXPECT_EQ(one.num_ops, 1);
  EXPECT_EQ(one.num_reads, 1);
  EXPECT_EQ(one.bytes_transferred, nbytes());
  EXPECT_EQ(one.bytes_read, nbytes());
  EXPECT_GT(one.busy, kvikio::Duration::zero());
  EXPECT_LE(one.busy, baseline.busy);
  EXPECT_LT(one.total_duration, baseline.total_duration);

  std::uint64_t ops   = 0;
  std::uint64_t bytes = 0;
  auto busy           = kvikio::Duration::zero();
  for (auto const& interval : intervals) {
    ops += interval.num_ops;
    bytes += interval.bytes_transferred;
    busy += interval.busy;
    EXPECT_LE(interval.busy, interval.wall());
  }
  // Nothing counted twice and nothing lost, in the counters or on the time axis.
  EXPECT_EQ(ops, 4);
  EXPECT_EQ(bytes, 4 * nbytes());
  EXPECT_EQ(busy, baseline.busy - first.busy);
  EXPECT_EQ(intervals.front().start, first.end);
  EXPECT_EQ(intervals.back().end, baseline.end);
  for (std::size_t i = 1; i < intervals.size(); ++i) {
    EXPECT_EQ(intervals[i].start, intervals[i - 1].end) << "the intervals abut";
  }
}

TEST_F(SummaryTest, a_reading_never_goes_backwards)
{
  Delayer delayer;
  // Registered ahead of ours, so a completion reaches us long after its end was stamped.
  auto const id = kvikio::register_monitor(&delayer);
  SummaryMonitor const monitor;

  auto recorder = std::make_unique<kvikio::detail::LogicalObservationRecorder>(
    IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes());
  std::this_thread::sleep_for(std::chrono::milliseconds{5});
  std::thread stalled{[&] { recorder->finish(nbytes()); }};

  // Taken while the operation is still open here, so it runs the stretch up to now, past the end
  // the completion will eventually report.
  std::this_thread::sleep_for(std::chrono::milliseconds{30});
  auto const during = monitor.get();
  stalled.join();
  kvikio::unregister_monitor(id);

  auto const after = monitor.get();
  EXPECT_GE(after.busy, during.busy) << "the stretch closed earlier than the reading ran it to";
  // Which is what keeps the interval between them from being negative and clamped away.
  EXPECT_EQ(after.since(during).busy, after.busy - during.busy);
}

TEST_F(SummaryTest, since_refuses_a_baseline_it_cannot_subtract)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor monitor;
  SummaryMonitor const other;
  kvikio::FileHandle f{_filepath, "r"};

  f.pread(buffer.data(), nbytes(), 0).get();
  auto const first = monitor.get();

  // An interval holds differences, not totals, so subtracting it from the totals would count the
  // operations before it a second time. A periodic reporter keeps the last `get()`, not the last
  // interval, and reaching for the wrong one says so.
  f.pread(buffer.data(), nbytes(), 0).get();
  auto const interval = monitor.since(first);
  EXPECT_EQ(interval.num_ops, 1);
  EXPECT_THROW(std::ignore = monitor.since(interval), std::invalid_argument);

  // Nor can a reading of one monitor be subtracted from another's.
  EXPECT_THROW(std::ignore = other.since(first), std::invalid_argument);

  // Nor a later reading from an earlier one, which shares the span and so passes on that alone.
  auto const later = monitor.get();
  EXPECT_THROW(std::ignore = first.since(later), std::invalid_argument);

  // A reset throws away the totals the earlier reading was taken against, leaving nothing to
  // subtract rather than an interval of zero.
  monitor.reset();
  EXPECT_THROW(std::ignore = monitor.since(first), std::invalid_argument);
}

TEST_F(SummaryTest, an_operation_still_running_counts_as_busy)
{
  // The point of measuring busy time as the work happens: an operation that has not finished has
  // delivered no observation, but the process is busy all the same.
  SummaryMonitor const monitor;
  {
    kvikio::detail::LogicalObservationRecorder const open{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes()};
    std::this_thread::sleep_for(std::chrono::milliseconds{20});

    auto const during = monitor.get();
    EXPECT_EQ(during.num_ops, 0) << "nothing has been reported yet";
    EXPECT_GE(during.busy, std::chrono::milliseconds{15})
      << "the open operation should already count";
    EXPECT_LE(during.busy, during.wall());
  }
  auto const after = monitor.get();
  EXPECT_EQ(after.num_ops, 1);
  EXPECT_GE(after.busy, std::chrono::milliseconds{20});
}

TEST_F(SummaryTest, a_finish_from_an_earlier_monitor_is_ignored)
{
  // A finish is delivered to whoever is registered when the operation ends, which need not be who
  // was registered when it began. A monitor created while somebody else's operation is still open
  // must not treat that operation's finish as the end of its own.
  auto older = std::make_unique<kvikio::detail::LogicalObservationRecorder>(
    IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes());

  SummaryMonitor const monitor;  // Created while `older` is still in flight.
  {
    kvikio::detail::LogicalObservationRecorder const mine{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes()};
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    older.reset();  // The stale finish lands here, in the middle of our own operation.
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }

  auto const s = monitor.get();
  // The older operation is ignored whole: we never saw it start, so we can neither time it nor
  // honestly count its bytes against a span we were not measuring.
  EXPECT_EQ(s.num_ops, 1) << "an operation that began before the monitor was counted";
  // Ours ran for ~20 ms and was open throughout. Had the older operation's completion closed our
  // stretch, busy time would have stopped at the halfway point and our own would have been dropped.
  EXPECT_GE(s.busy, std::chrono::milliseconds{18});
  EXPECT_LE(s.busy, s.wall());
}

TEST_F(SummaryTest, a_stretch_closes_at_the_last_operation_to_end)
{
  // Completions are serialized by the monitor's lock, in whatever order threads reach it, which
  // is not necessarily the order their end timestamps were stamped in. Driving the notifications
  // directly is the only way to pin the case down.
  SummaryMonitor const monitor;
  auto const t0 = kvikio::detail::now() + std::chrono::nanoseconds{1};

  Observation first{};
  first.start = t0;
  first.end   = t0 + std::chrono::nanoseconds{50};
  Observation second{};
  second.start = t0 + std::chrono::nanoseconds{1};
  second.end   = t0 + std::chrono::nanoseconds{100};  // Ends later, but reports first.

  kvikio::detail::notify_started(first);
  kvikio::detail::notify_started(second);
  kvikio::detail::notify_finished(second);
  kvikio::detail::notify_finished(first);

  // The stretch ran until the last of its operations ended, at t0+100. Closing it at whichever
  // completion happened to arrive last would have dropped the 50 ns tail.
  auto const s = monitor.get();
  EXPECT_EQ(s.num_ops, 2);
  EXPECT_GE(s.busy, std::chrono::nanoseconds{100}) << "the stretch was closed at the earlier end";
}

TEST_F(SummaryTest, a_start_reported_late_extends_the_stretch_backwards)
{
  // The mirror of the case above, on the opening side: starts are serialized by the monitor's lock
  // in whatever order threads reach it, which is not the order they were stamped in.
  SummaryMonitor const monitor;
  auto const t0 = kvikio::detail::now() + std::chrono::nanoseconds{1};

  Observation first{};
  first.start = t0 + std::chrono::nanoseconds{50};  // Begins later, but reports first.
  first.end   = t0 + std::chrono::nanoseconds{100};
  Observation second{};
  second.start = t0;
  second.end   = t0 + std::chrono::nanoseconds{100};

  kvikio::detail::notify_started(first);
  kvikio::detail::notify_started(second);
  kvikio::detail::notify_finished(first);
  kvikio::detail::notify_finished(second);

  // The stretch began with the earliest of its operations, at t0. Anchoring it at the first start
  // to arrive would have dropped the 50 ns that only the second one covered.
  auto const s = monitor.get();
  EXPECT_EQ(s.num_ops, 2);
  EXPECT_GE(s.busy, std::chrono::nanoseconds{100}) << "the stretch was anchored at the later start";
}

TEST_F(SummaryTest, a_record_from_before_a_reset_is_ignored)
{
  // An operation reports its finish before its record is delivered, so a `reset()` can land
  // between the two. The record must not then add operations to a span that has no time for them.
  SummaryMonitor monitor;
  auto const before_reset = kvikio::detail::now();
  std::this_thread::sleep_for(std::chrono::milliseconds{5});
  monitor.reset();

  Observation stale{};
  stale.start             = before_reset;
  stale.end               = before_reset + std::chrono::nanoseconds{1000};
  stale.size              = nbytes();
  stale.bytes_transferred = nbytes();
  kvikio::detail::notify_finished(stale);

  EXPECT_EQ(monitor.get().num_ops, 0)
    << "an operation that finished before the reset was counted after it";
  EXPECT_EQ(monitor.get().bytes_transferred, 0);

  // `busy` only ever holds the part of an in-flight operation that followed the reset, and the
  // counters have to agree, or the bytes and duration of work that ran before the span would be
  // divided by the time since it.
  {
    kvikio::detail::LogicalObservationRecorder const spanning{
      IoBackend::POSIX, TransferDirection::READ, MemoryKind::HOST, 0, nbytes()};
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    monitor.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }

  auto const s = monitor.get();
  EXPECT_EQ(s.num_ops, 0) << "an operation that began before the reset was counted after it";
  EXPECT_EQ(s.bytes_transferred, 0);
  EXPECT_EQ(s.total_duration, kvikio::Duration::zero());
  EXPECT_LE(s.busy, s.wall());
}

TEST_F(SummaryTest, json_survives_a_locale_with_a_decimal_comma)
{
  // `to_json()` promises something a parser will accept, whatever the application set as its
  // global locale.
  struct DecimalComma : std::numpunct<char> {
    [[nodiscard]] char do_decimal_point() const override { return ','; }
  };

  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }

  auto const previous = std::locale::global(std::locale{std::locale::classic(), new DecimalComma});
  auto const json     = monitor.get().to_json();
  std::locale::global(previous);

  EXPECT_NE(json.find("\"busy_fraction\": 0."), std::string::npos) << json;
}

TEST_F(SummaryTest, stop_freezes_the_totals_and_the_span)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor monitor;
  kvikio::FileHandle f{_filepath, "r"};

  f.pread(buffer.data(), nbytes(), 0).get();
  monitor.stop();
  auto const first = monitor.get();
  EXPECT_EQ(first.num_ops, 1);
  ASSERT_GT(first.busy, kvikio::Duration::zero());

  f.pread(buffer.data(), nbytes(), 0).get();
  std::this_thread::sleep_for(std::chrono::milliseconds{30});
  monitor.stop();  // Idempotent.
  auto const second = monitor.get();

  EXPECT_EQ(second.num_ops, first.num_ops) << "counting continued after stop()";
  // The measured span ended at `stop()` too. Were `end` stamped per reading, the wall time would
  // have grown by the sleep and `busy_fraction()` would have decayed with it.
  EXPECT_EQ(second.end, first.end);
  EXPECT_EQ(second.wall(), first.wall());
  EXPECT_DOUBLE_EQ(second.busy_fraction(), first.busy_fraction());
  EXPECT_EQ(monitor.since(first).end, first.end);
}

TEST_F(SummaryTest, stopping_from_several_threads_at_once_is_safe)
{
  // A monitor shared between threads can be stopped by any of them, and the totals are final once
  // any one of the calls returns.
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }

  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&monitor] { monitor.stop(); });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  auto const s = monitor.get();
  EXPECT_EQ(s.num_ops, 1);
  EXPECT_EQ(s.bytes_transferred, nbytes());
  EXPECT_LE(s.busy, s.wall());
}

TEST_F(SummaryTest, a_monitor_reports_itself_on_destruction)
{
  std::vector<std::uint64_t> buffer(_data.size());
  Summary reported;
  {
    SummaryMonitor const monitor{[&reported](Summary const& s) { reported = s; }};
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  // The callback runs after counting has stopped, so the totals it receives are final.
  EXPECT_EQ(reported.num_ops, 1);
  EXPECT_EQ(reported.bytes_transferred, nbytes());

  EXPECT_NO_THROW(
    { SummaryMonitor const monitor{[](Summary const&) { throw std::runtime_error{"boom"}; }}; });
}

TEST_F(SummaryTest, a_summary_survives_a_round_trip_through_bytes)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }

  auto const before = monitor.get();
  auto const bytes  = before.serialize();
  auto const after  = Summary::deserialize(bytes);
  // Serialising again gives the same bytes, so every field that goes in comes back.
  EXPECT_EQ(after.serialize(), bytes);
  EXPECT_EQ(after.report(), before.report());
  // The anchor survives, which JSON's wall-clock projection could not carry.
  EXPECT_EQ(after.to_json(), before.to_json());
  EXPECT_EQ(after.start, before.start);
  EXPECT_EQ(after.end, before.end);

  // Still a reading of this monitor, so an interval can be measured from it.
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  EXPECT_EQ(monitor.since(after).num_ops, 1);
}

TEST_F(SummaryTest, bytes_that_are_not_a_summary_are_refused)
{
  SummaryMonitor const monitor;
  auto const good = monitor.get().serialize();

  EXPECT_THROW(std::ignore = Summary::deserialize({}), std::invalid_argument);

  auto truncated = good;
  truncated.pop_back();
  EXPECT_THROW(std::ignore = Summary::deserialize(truncated), std::invalid_argument);

  auto not_ours = good;
  not_ours[0]   = std::byte{'X'};
  EXPECT_THROW(std::ignore = Summary::deserialize(not_ours), std::invalid_argument);

  // Summaries do not cross versions, and saying so beats reading the fields as if they matched.
  auto from_the_future = good;
  from_the_future[4]   = std::byte{99};
  EXPECT_THROW(std::ignore = Summary::deserialize(from_the_future), std::invalid_argument);

  // Nor do they cross byte orders. The header is readable either way, so without the mark the
  // payload would be reinterpreted rather than refused.
  auto other_endian = good;
  std::swap(other_endian[12], other_endian[15]);
  std::swap(other_endian[13], other_endian[14]);
  EXPECT_THROW(std::ignore = Summary::deserialize(other_endian), std::invalid_argument);
}

TEST_F(SummaryTest, a_summary_says_which_observations_it_is_over)
{
  // The kind labels the monitor, so it holds whether or not anything was counted.
  SummaryMonitor const calls;
  SummaryMonitor transfers{kvikio::ObservationKind::PHYSICAL};
  auto const logical  = calls.get();
  auto const physical = transfers.get();

  EXPECT_EQ(logical.kind, kvikio::ObservationKind::LOGICAL);
  EXPECT_EQ(physical.kind, kvikio::ObservationKind::PHYSICAL);
  EXPECT_NE(logical.report().find("(LOGICAL)"), std::string::npos);
  EXPECT_NE(physical.report().find("(PHYSICAL)"), std::string::npos);
  EXPECT_NE(physical.to_json().find("\"kind\": \"PHYSICAL\""), std::string::npos);

  // It survives everything that carries totals forward.
  EXPECT_EQ(transfers.since(physical).kind, kvikio::ObservationKind::PHYSICAL);
  EXPECT_EQ(Summary::deserialize(physical.serialize()).kind, kvikio::ObservationKind::PHYSICAL);
  transfers.reset();
  EXPECT_EQ(transfers.get().kind, kvikio::ObservationKind::PHYSICAL);
}

TEST_F(SummaryTest, the_internal_costs_are_those_of_the_span)
{
  // Everything counted is remote, so a run against a local file owes nothing and says so by
  // leaving the rows out.
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }

  auto const s = monitor.get();
  EXPECT_TRUE(s.counters.empty()) << "a local read owes none of what is counted";
  EXPECT_EQ(s.report().find("size probes"), std::string::npos);
  // The JSON schema does not vary, unlike the report.
  EXPECT_NE(s.to_json().find("\"counters\""), std::string::npos);

  monitor.reset();
  EXPECT_TRUE(monitor.get().counters.empty());
}

TEST_F(SummaryTest, the_report_is_human_readable)
{
  std::vector<std::uint64_t> buffer(_data.size());
  SummaryMonitor const monitor;
  {
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  auto const text = monitor.get().report();

  EXPECT_NE(text.find("KvikIO I/O summary"), std::string::npos);
  for (auto const* label : {"wall time",
                            "busy time",
                            "busy bandwidth",
                            "operations",
                            "mean duration",
                            "bytes",
                            "errors"}) {
    EXPECT_NE(text.find(label), std::string::npos) << "missing row: " << label;
  }
  EXPECT_NE(text.find("MiB"), std::string::npos) << "byte counts are scaled";
  EXPECT_NE(text.find("% of the wall time"), std::string::npos);

  auto const json = monitor.get().to_json();
  EXPECT_NE(json.find("\"num_ops\""), std::string::npos);
  // Wall clock, and named for it, so a reader can line the summary up with another log.
  EXPECT_NE(json.find("\"start_unix_ns\""), std::string::npos);
  EXPECT_EQ(json.find("\"start_ns\""), std::string::npos);
  EXPECT_NE(json.find("\"busy_ns\""), std::string::npos);
}
