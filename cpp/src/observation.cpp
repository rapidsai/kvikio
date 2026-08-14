/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/observation.hpp>

namespace kvikio {

namespace {

/// Number of registered monitors. The gate `monitoring_enabled()` reads.
std::atomic<std::uint32_t> monitor_count{0};

/**
 * @brief The registered monitors, the lock over them, and the notification path.
 */
class Registry {
 public:
  /**
   * @brief The one registry.
   *
   * Intentionally leaked: I/O can complete during static destruction, for instance on a detached
   * remote-I/O reactor thread, and must still find a valid registry.
   *
   * @return The registry.
   */
  static Registry& instance()
  {
    static auto* registry = new Registry{};
    return *registry;
  }

  /**
   * @brief Add a monitor.
   *
   * @param monitor The monitor. Not owned.
   * @param kind Which observations it watches.
   * @return The new registration id.
   */
  std::uint64_t add(Monitor* monitor, ObservationKind kind)
  {
    KVIKIO_EXPECT(monitor != nullptr, "the monitor must not be null", std::invalid_argument);
    KVIKIO_EXPECT(!t_in_monitor,
                  "cannot register a monitor from inside a monitor callback",
                  std::runtime_error);

    auto const id = _next_id.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock const lock{_mutex};
    _monitors.push_back(Entry{id, kind, monitor});
    publish_count();
    return id;
  }

  /**
   * @brief Remove a monitor, waiting for any in-flight call to it to finish.
   *
   * @param id The registration id. Unknown ids are ignored.
   */
  void remove(std::uint64_t id)
  {
    // The notification path holds `_mutex` shared for the duration of the callbacks, so taking it
    // exclusively from inside one would wait for the caller's own lock to be released.
    KVIKIO_EXPECT(!t_in_monitor,
                  "cannot unregister a monitor from inside a monitor callback",
                  std::runtime_error);

    // Taking the mutex exclusively waits for every in-flight notification to finish, so the caller
    // may destroy the monitor as soon as this returns.
    std::unique_lock const lock{_mutex};
    auto const it =
      std::find_if(_monitors.begin(), _monitors.end(), [id](auto const& e) { return e.id == id; });
    if (it == _monitors.end()) { return; }
    _monitors.erase(it);
    publish_count();
  }

  /// Tell the monitors subscribed to its kind that an operation started.
  void started(Observation const& observation) noexcept { notify(observation, Phase::Start); }

  /// Tell the monitors subscribed to its kind that an operation finished.
  void finished(Observation const& observation) noexcept { notify(observation, Phase::Finish); }

  /**
   * @brief Whether the calling thread is currently executing a monitor callback.
   *
   * @return True if inside a callback.
   */
  [[nodiscard]] static bool in_monitor() noexcept { return t_in_monitor; }

 private:
  enum class Phase : std::uint8_t { Start, Finish };

  void notify(Observation const& observation, Phase phase) noexcept
  {
    // Fast path: with nobody registered we never reach the mutex.
    if (!detail::monitoring_enabled()) { return; }
    // Belt and braces: a monitor is forbidden from calling into KvikIO, but should one manage it
    // through a path that does not check, do not deliver the observation back into the monitors,
    // which would recurse and re-enter the shared lock.
    if (t_in_monitor) { return; }

    std::shared_lock const lock{_mutex};
    InMonitorGuard const guard;
    for (auto const& entry : _monitors) {
      // Subscription is per kind, so a layer that starts emitting later cannot flood a monitor that
      // never asked for it.
      if (entry.kind != observation.kind) { continue; }
      if (phase == Phase::Start) {
        entry.monitor->on_start(observation);
      } else {
        entry.monitor->on_finish(observation);
      }
    }
  }

  /// Republish the count. Must be called with `_mutex` held exclusively.
  void publish_count() const noexcept
  {
    monitor_count.store(static_cast<std::uint32_t>(_monitors.size()), std::memory_order_release);
  }

  /**
   * @brief Whether this thread is currently executing a monitor callback.
   *
   * Guards against two ways a monitor can hang the process. One that performs KvikIO I/O would emit
   * an observation and recursively acquire the shared lock, which `std::shared_mutex` does not
   * support, so such observations are dropped instead. And one that unregisters would block
   * forever waiting for itself, so that is rejected outright.
   */
  static inline thread_local bool t_in_monitor{false};

  /// RAII guard for `t_in_monitor`.
  class InMonitorGuard {
   public:
    InMonitorGuard() noexcept { t_in_monitor = true; }
    ~InMonitorGuard() { t_in_monitor = false; }
    InMonitorGuard(InMonitorGuard const&)            = delete;
    InMonitorGuard& operator=(InMonitorGuard const&) = delete;
  };

  /// One registered monitor. Not owned, and must outlive its registration.
  struct Entry {
    std::uint64_t id{0};
    ObservationKind kind{ObservationKind::Logical};
    Monitor* monitor{nullptr};
  };

  std::shared_mutex _mutex;
  std::vector<Entry> _monitors;
  std::atomic<std::uint64_t> _next_id{1};
};

}  // namespace

namespace detail {

bool monitoring_enabled() noexcept { return monitor_count.load(std::memory_order_acquire) != 0; }
}  // namespace detail

std::string_view to_string(IoBackend backend) noexcept
{
  switch (backend) {
    case IoBackend::Posix: return "Posix";
    case IoBackend::Gds: return "Gds";
    case IoBackend::Mmap: return "Mmap";
    case IoBackend::RemoteHttp: return "RemoteHttp";
    case IoBackend::RemoteHdfs: return "RemoteHdfs";
    default: return "Unknown";
  }
}

std::string_view to_string(TransferDirection direction) noexcept
{
  switch (direction) {
    case TransferDirection::Read: return "Read";
    case TransferDirection::Write: return "Write";
    default: return "Unknown";
  }
}

std::string_view to_string(MemoryKind memory_kind) noexcept
{
  switch (memory_kind) {
    case MemoryKind::Host: return "Host";
    case MemoryKind::Device: return "Device";
    default: return "Unknown";
  }
}

std::string_view to_string(ObservationKind kind) noexcept
{
  switch (kind) {
    case ObservationKind::Logical: return "Logical";
    default: return "Unknown";
  }
}

std::uint64_t register_monitor(Monitor* monitor, ObservationKind kind)
{
  return Registry::instance().add(monitor, kind);
}

void unregister_monitor(std::uint64_t id) { Registry::instance().remove(id); }

namespace detail {

void expect_not_in_monitor()
{
  KVIKIO_EXPECT(
    !Registry::in_monitor(), "a monitor must not call back into KvikIO", std::runtime_error);
}

void LogicalObservationRecorder::begin(IoBackend backend,
                                       TransferDirection direction,
                                       MemoryKind memory_kind,
                                       std::size_t offset,
                                       std::size_t size,
                                       char const* http_method) noexcept
{
  // From 1, so that a default-constructed `Observation` (id 0) is never mistaken for a real one.
  static std::atomic<std::uint64_t> id_counter{1};
  _observation.backend     = backend;
  _observation.direction   = direction;
  _observation.memory_kind = memory_kind;
  _observation.offset      = offset;
  _observation.size        = size;
  _observation.http_method = http_method;
  _observation.id          = id_counter.fetch_add(1, std::memory_order_relaxed);
  _observation.start       = now();
  if (monitor_count.load(std::memory_order_acquire) != 0) { notify_started(_observation); }
}

void LogicalObservationRecorder::emit() noexcept
{
  _observation.end = now();
  // A monitor registered after this operation began never saw its start and ignores the finish,
  // which is why no flag is latched here.
  notify_finished(_observation);
}

void notify_started(Observation const& observation) noexcept
{
  Registry::instance().started(observation);
}

void notify_finished(Observation const& observation) noexcept
{
  Registry::instance().finished(observation);
}

}  // namespace detail

}  // namespace kvikio
