/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/mmap.hpp>
#include <kvikio/observation.hpp>
#ifdef KVIKIO_LIBCURL_FOUND
#include <kvikio/remote_handle.hpp>
#endif

#include "utils/utils.hpp"

using kvikio::IoBackend;
using kvikio::MemoryKind;
using kvikio::Observation;
using kvikio::ObservationKind;
using kvikio::TransferDirection;

namespace {

/// A monitor that keeps every observation it is given, for the duration of a `Capture`.
class Recorder final : public kvikio::Monitor {
 public:
  static Recorder& instance()
  {
    static Recorder recorder;
    return recorder;
  }

  void on_start(Observation const&) noexcept override {}

  void on_finish(Observation const& observation) noexcept override
  {
    std::lock_guard const lock{_mutex};
    if (_capturing) { _observations.push_back(observation); }
  }

  /**
   * @brief RAII: register the recorder and capture observations, for the duration of a test.
   */
  class Capture {
   public:
    Capture()
    {
      auto& self = Recorder::instance();
      {
        std::lock_guard const lock{self._mutex};
        self._observations.clear();
        self._capturing = true;
      }
      self._id = kvikio::register_monitor(&self);
    }
    ~Capture()
    {
      auto& self = Recorder::instance();
      kvikio::unregister_monitor(self._id);
      std::lock_guard const lock{self._mutex};
      self._capturing = false;
    }
    Capture(Capture const&)            = delete;
    Capture& operator=(Capture const&) = delete;
  };

  [[nodiscard]] std::vector<Observation> observations() const
  {
    std::lock_guard const lock{_mutex};
    return _observations;
  }

 private:
  mutable std::mutex _mutex;
  std::uint64_t _id{0};
  bool _capturing{false};
  std::vector<Observation> _observations;
};

}  // namespace

/// A monitor that runs a callable on finish, so a test can express one inline.
class CallbackMonitor final : public kvikio::Monitor {
 public:
  explicit CallbackMonitor(std::function<void(Observation const&)> on_finish)
    : _on_finish{std::move(on_finish)}
  {
    _id = kvikio::register_monitor(this);
  }
  ~CallbackMonitor() override { kvikio::unregister_monitor(_id); }
  CallbackMonitor(CallbackMonitor const&)            = delete;
  CallbackMonitor& operator=(CallbackMonitor const&) = delete;

 private:
  void on_start(Observation const&) noexcept override {}
  void on_finish(Observation const& observation) noexcept override { _on_finish(observation); }

  std::function<void(Observation const&)> _on_finish;
  std::uint64_t _id{0};
};

class ObservationTest : public testing::Test {
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

TEST(ObservationBasics, enum_names)
{
  EXPECT_EQ(to_string(IoBackend::POSIX), "POSIX");
  EXPECT_EQ(to_string(IoBackend::MMAP), "MMAP");
  EXPECT_EQ(to_string(TransferDirection::WRITE), "WRITE");
  EXPECT_EQ(to_string(MemoryKind::DEVICE), "DEVICE");
  EXPECT_EQ(to_string(ObservationKind::LOGICAL), "LOGICAL");
}

TEST(ObservationBasics, derived_quantities)
{
  Observation o{};
  o.start             = kvikio::TimePoint{std::chrono::nanoseconds{1'000}};
  o.end               = kvikio::TimePoint{std::chrono::nanoseconds{3'000}};
  o.bytes_transferred = 4096;
  EXPECT_EQ(o.duration(), std::chrono::nanoseconds{2'000});
  EXPECT_DOUBLE_EQ(o.bytes_per_sec(), 4096.0 * 1e9 / 2000.0);

  Observation const empty{};
  EXPECT_EQ(empty.duration(), kvikio::Duration::zero());
  EXPECT_EQ(empty.bytes_per_sec(), 0.0);
}

TEST(ObservationBasics, an_anchor_maps_the_clock_onto_the_wall_clock)
{
  // The two clocks are read as close together as the platform allows, so the anchor's own reading
  // maps back onto itself.
  auto const anchor = kvikio::ClockAnchor::now();
  EXPECT_EQ(anchor.to_wall_clock(anchor.steady), anchor.wall);

  // An offset on one clock is the same offset on the other.
  auto const later = anchor.steady + std::chrono::seconds{5};
  EXPECT_EQ(anchor.to_wall_clock(later) - anchor.wall, std::chrono::seconds{5});
  auto const earlier = anchor.steady - std::chrono::milliseconds{250};
  EXPECT_EQ(anchor.wall - anchor.to_wall_clock(earlier), std::chrono::milliseconds{250});

  // An observation's timestamps land near the wall-clock time of the run, within the second it
  // took to get here.
  auto const now = std::chrono::system_clock::now();
  EXPECT_LT(anchor.to_wall_clock(kvikio::detail::now()) - now, std::chrono::seconds{1});
}

TEST(ObservationBasics, a_null_monitor_is_rejected)
{
  EXPECT_THROW(std::ignore = kvikio::register_monitor(nullptr), std::invalid_argument);
}

TEST(ObservationBasics, nothing_is_emitted_without_a_monitor)
{
  EXPECT_FALSE(kvikio::detail::monitoring_enabled());
}

TEST_F(ObservationTest, one_call_is_one_observation)
{
  std::vector<std::uint64_t> buffer(_data.size());
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  auto const events = Recorder::instance().observations();

  // However many reads KvikIO issued underneath, the caller made one call.
  ASSERT_EQ(events.size(), 1);
  auto const& o = events.front();
  EXPECT_EQ(o.kind, ObservationKind::LOGICAL);
  EXPECT_EQ(o.direction, TransferDirection::READ);
  EXPECT_EQ(o.memory_kind, MemoryKind::HOST);
  // A host buffer goes through `posix_host_read` whatever the handle is capable of, so the label
  // must not follow the handle's compatibility mode.
  EXPECT_EQ(o.backend, IoBackend::POSIX);
  EXPECT_EQ(o.offset, 0);
  EXPECT_EQ(o.size, nbytes());
  EXPECT_EQ(o.bytes_transferred, nbytes());
  EXPECT_TRUE(o.ok);
  EXPECT_NE(o.id, 0);
  EXPECT_GT(o.end, o.start);
}

TEST_F(ObservationTest, a_rejected_call_is_not_an_observation)
{
  std::vector<std::uint64_t> buffer(_data.size());
  Recorder::Capture const capture;
  kvikio::FileHandle f{_filepath, "r"};

  // Rejected on its arguments, so it never reaches the file. An operation that did nothing is not
  // one the monitors should be told about, in either direction.
  EXPECT_THROW(f.pread(buffer.data(), nbytes(), 0, 0).get(), std::invalid_argument);
  EXPECT_THROW(f.pwrite(buffer.data(), nbytes(), 0, 0).get(), std::invalid_argument);
  EXPECT_TRUE(Recorder::instance().observations().empty());
}

TEST_F(ObservationTest, an_observation_names_what_it_read)
{
  Recorder::Capture const capture;
  std::vector<std::uint64_t> buffer(_data.size());
  kvikio::FileHandle f{_filepath, "r"};
  f.pread(buffer.data(), nbytes(), 0).get();

  // Read while the handle is alive, since the handle owns the string the view points at.
  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 1);
  // Without this a program reading many files cannot tell their operations apart.
  EXPECT_EQ(events.front().source, _filepath);
}

TEST_F(ObservationTest, a_moved_handle_still_names_what_it_reads)
{
  Recorder::Capture const capture;
  std::vector<std::uint64_t> buffer(_data.size());

  kvikio::MmapHandle moved_from{_filepath, "r"};
  kvikio::MmapHandle mapped{std::move(moved_from)};
  mapped.read(buffer.data(), nbytes(), 0);

  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().source, _filepath) << "the path did not survive the move";
}

TEST_F(ObservationTest, the_span_covers_the_work_not_the_call)
{
  // `pread()` returns as soon as the parts are submitted, so a naive recorder would stop the clock
  // far too early. The observation must cover the I/O, and must already be delivered by the time
  // the caller's future is ready.
  std::vector<std::uint64_t> buffer(_data.size());
  kvikio::TimePoint after_submit{};
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath, "r"};
    auto future  = f.pread(buffer.data(), nbytes(), 0);
    after_submit = kvikio::detail::now();
    future.get();

    auto const events = Recorder::instance().observations();
    ASSERT_EQ(events.size(), 1) << "delivered before the future became ready";
    EXPECT_GT(events.front().end, after_submit) << "the clock ran past the submit";
  }
}

TEST_F(ObservationTest, ids_are_unique_per_call)
{
  std::vector<std::uint64_t> buffer(_data.size());
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
    f.pread(buffer.data(), nbytes(), 0).get();
  }

  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 2);
  EXPECT_NE(events[0].id, events[1].id);
}

TEST_F(ObservationTest, a_write_is_described_as_a_write)
{
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath + ".w", "w"};
    f.pwrite(_data.data(), nbytes(), 0).get();
  }
  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().direction, TransferDirection::WRITE);
  EXPECT_EQ(events.front().bytes_transferred, nbytes());
}

TEST_F(ObservationTest, an_mmap_read_is_tagged_mmap)
{
  std::uint64_t value{0};
  {
    Recorder::Capture const capture;
    kvikio::MmapHandle m{_filepath, "r"};
    m.read(&value, sizeof(value), 0);
  }
  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().backend, IoBackend::MMAP);
  EXPECT_EQ(events.front().bytes_transferred, sizeof(value));
}

TEST_F(ObservationTest, a_failed_call_is_recorded_as_failed)
{
  {
    Recorder::Capture const capture;
    // Opened read-only, so the write fails and the failure must surface as one failed logical
    // operation rather than being lost.
    kvikio::FileHandle f{_filepath, "r"};
    EXPECT_ANY_THROW(f.pwrite(_data.data(), nbytes(), 0).get());
  }
  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 1);
  EXPECT_FALSE(events.front().ok);
  EXPECT_EQ(events.front().bytes_transferred, 0);
}

TEST_F(ObservationTest, a_failed_blocking_call_is_recorded_as_failed)
{
  // The scope-bound use of the recorder: nobody calls `finish()`, so the destructor does it, and
  // it has to notice that the scope is being left by an exception.
  kvikio::test::DevBuffer<std::uint64_t> const dev{_data};
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath, "r"};  // Read-only, so the write cannot succeed.
    EXPECT_ANY_THROW(f.write(dev.ptr, nbytes(), 0, 0));
  }
  auto const events = Recorder::instance().observations();
  ASSERT_EQ(events.size(), 1);
  EXPECT_FALSE(events.front().ok);
  EXPECT_EQ(events.front().bytes_transferred, 0);
}

TEST_F(ObservationTest, a_monitor_that_calls_into_kvikio_is_rejected)
{
  std::vector<std::uint64_t> buffer(_data.size());
  std::atomic<int> rejected{0};
  {
    CallbackMonitor const monitor{[&](Observation const&) {
      try {
        kvikio::FileHandle f{_filepath, "r"};
        std::vector<std::uint64_t> nested(16);
        f.pread(nested.data(), nested.size() * sizeof(std::uint64_t), 0).get();
      } catch (std::runtime_error const&) {
        ++rejected;
      }
    }};
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  EXPECT_GT(rejected.load(), 0);
}

TEST_F(ObservationTest, unregistering_waits_for_an_in_flight_monitor)
{
  std::vector<std::uint64_t> buffer(_data.size());
  auto state = std::make_shared<std::atomic<int>>(0);
  {
    CallbackMonitor const monitor{[state](Observation const&) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      ++*state;
    }};
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  // Once the monitor is gone, no thread can still be inside it.
  EXPECT_EQ(state.use_count(), 1);
}

TEST_F(ObservationTest, a_monitor_can_correlate_a_start_with_its_completion)
{
  // Written against the public interface only: this is what a user's monitor looks like. It keys
  // per-operation state by `Observation::id`, which is the reason the start carries the record
  // rather than a bare timestamp, because two operations can be stamped in the same nanosecond.
  class Correlator final : public kvikio::Monitor {
   public:
    void on_start(Observation const& o) noexcept override
    {
      std::lock_guard const lock{_mutex};
      _open[o.id] = o;
      // What is known at submission is set. What `on_start()` documents as not yet set, is not.
      if (o.size == 0 || o.id == 0) { ++_malformed; }
      if (o.end != kvikio::TimePoint{} || o.bytes_transferred != 0) { ++_malformed; }
    }

    void on_finish(Observation const& o) noexcept override
    {
      std::lock_guard const lock{_mutex};
      auto const it = _open.find(o.id);
      if (it == _open.end()) {
        ++_unmatched;
        return;
      }
      // The completion carries the same operation, now with an end.
      if (it->second.size == o.size && it->second.start == o.start && o.end > o.start) {
        ++_matched;
      }
      _open.erase(it);
    }

    std::mutex _mutex;
    std::unordered_map<std::uint64_t, Observation> _open;
    int _matched{0};
    int _unmatched{0};
    int _malformed{0};
  };

  Correlator correlator;
  auto const id = kvikio::register_monitor(&correlator);
  {
    kvikio::FileHandle f{_filepath, "r"};
    std::vector<std::uint64_t> buffer(_data.size());
    f.pread(buffer.data(), nbytes(), 0).get();
    f.pread(buffer.data(), nbytes(), 0).get();
  }
  kvikio::unregister_monitor(id);  // Blocks, so nothing is inside `correlator` after this.

  EXPECT_EQ(correlator._matched, 2) << "a completion did not find its start";
  EXPECT_EQ(correlator._unmatched, 0);
  EXPECT_EQ(correlator._malformed, 0) << "the record at start was not as documented";
  EXPECT_TRUE(correlator._open.empty()) << "an operation started and never completed";
}

TEST_F(ObservationTest, unregistering_a_monitor_from_a_callback_is_rejected)
{
  // The notification path holds the registry lock shared while a monitor runs, so taking it
  // exclusively from inside would wait on the caller's own lock.
  class Noop final : public kvikio::Monitor {
   public:
    void on_start(Observation const&) noexcept override {}
    void on_finish(Observation const&) noexcept override {}
  };
  Noop noop;
  auto const monitor_id = kvikio::register_monitor(&noop);

  std::atomic<int> rejected{0};
  {
    CallbackMonitor const canary{[&](Observation const&) {
      try {
        kvikio::unregister_monitor(monitor_id);
      } catch (std::runtime_error const&) {
        ++rejected;
      }
    }};
    kvikio::MmapHandle m{_filepath, "r"};
    std::uint64_t sink{0};
    m.read(&sink, sizeof(sink), 0);
  }
  EXPECT_EQ(rejected.load(), 1) << "the re-entrant removal was not rejected";

  kvikio::unregister_monitor(monitor_id);  // Still registered, and removable from outside.
}

TEST_F(ObservationTest, a_sub_threshold_device_call_is_observed_as_posix)
{
  // Below the GDS threshold `pread()` skips the thread pool and reads inline through
  // `posix_device_read`, a path that never reaches `parallel_io` and so has to close the
  // observation itself.
  kvikio::test::DevBuffer<std::uint64_t> const dev{_data.size()};
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(dev.ptr,
            nbytes(),
            0,
            kvikio::defaults::task_size(),
            /* gds_threshold = */ nbytes() + 1)
      .get();
  }
  auto const events = Recorder::instance().observations();

  ASSERT_EQ(events.size(), 1);
  auto const& o = events.front();
  EXPECT_EQ(o.backend, IoBackend::POSIX);
  EXPECT_EQ(o.memory_kind, MemoryKind::DEVICE);
  EXPECT_EQ(o.size, nbytes());
  EXPECT_EQ(o.bytes_transferred, nbytes()) << "the inline path reported no bytes";
  EXPECT_TRUE(o.ok);
}

TEST_F(ObservationTest, a_device_call_is_one_observation_too)
{
  // The device path fans the call out over the thread pool, and each worker performs a whole
  // `read()` of its own part. Those must not surface as observations in their own right: the
  // caller made one call, of the full size.
  kvikio::test::DevBuffer<std::uint64_t> const dev{_data.size()};
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath, "r"};
    f.pread(dev.ptr, nbytes(), 0).get();
  }
  auto const events = Recorder::instance().observations();

  ASSERT_EQ(events.size(), 1);
  auto const& o = events.front();
  EXPECT_EQ(o.memory_kind, MemoryKind::DEVICE);
  EXPECT_EQ(o.direction, TransferDirection::READ);
  EXPECT_EQ(o.size, nbytes());
  EXPECT_EQ(o.bytes_transferred, nbytes());
  EXPECT_TRUE(o.ok);
}

#ifdef KVIKIO_LIBCURL_FOUND
TEST_F(ObservationTest, a_remote_read_rejected_on_its_arguments_is_not_observed)
{
  // Opened with an explicit size and never contacted, so the read is rejected on its arguments
  // before any I/O is attempted. A call that never became an operation must not be reported as a
  // failed one.
  Recorder::Capture const capture;
  auto handle = kvikio::RemoteHandle::open(
    "http://127.0.0.1:1/nothing", kvikio::RemoteEndpointType::HTTP, std::nullopt, 8);
  std::uint64_t sink{0};

  EXPECT_THROW(handle.read(&sink, 1024, 0), std::invalid_argument);
  // `pread()` rejects it at the call too, on either backend, rather than through the future.
  EXPECT_THROW(std::ignore = handle.pread(&sink, 1024, 0), std::invalid_argument);
  // In bounds, but rejected on the task size, which is checked on both backends before anything
  // is submitted.
  EXPECT_THROW(std::ignore = handle.pread(&sink, 8, 0, 0), std::invalid_argument);
  EXPECT_TRUE(Recorder::instance().observations().empty());
}
#endif

TEST_F(ObservationTest, a_device_write_is_one_observation_too)
{
  kvikio::test::DevBuffer<std::uint64_t> const dev{_data};
  {
    Recorder::Capture const capture;
    kvikio::FileHandle f{_filepath + ".w", "w"};
    f.pwrite(dev.ptr, nbytes(), 0).get();
  }
  auto const events = Recorder::instance().observations();

  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().memory_kind, MemoryKind::DEVICE);
  EXPECT_EQ(events.front().direction, TransferDirection::WRITE);
  EXPECT_EQ(events.front().bytes_transferred, nbytes());
}
