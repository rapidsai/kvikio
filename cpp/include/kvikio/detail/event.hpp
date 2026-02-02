/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {
class EventPool {
 public:
  class Event {
   private:
    CUevent _event{nullptr};
    CUcontext _context{nullptr};

   public:
    explicit Event(CUevent event, CUcontext context) noexcept;
    ~Event() noexcept;

    // Move-only
    Event(Event const&)            = delete;
    Event& operator=(Event const&) = delete;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;

    [[nodiscard]] CUevent get() const noexcept;

    [[nodiscard]] CUcontext context() const noexcept;

    void record(CUstream stream);

    void synchronize();
  };

 private:
  std::mutex mutable _mutex;
  std::unordered_map<CUcontext, std::vector<CUevent>> _pools;

  EventPool() = default;

  // Intentionally leak events during static destruction. \sa BounceBufferPool
  ~EventPool() noexcept = default;

 public:
  // Non-copyable, non-movable singleton
  EventPool(EventPool const&)            = delete;
  EventPool& operator=(EventPool const&) = delete;
  EventPool(EventPool&&)                 = delete;
  EventPool& operator=(EventPool&&)      = delete;

  [[nodiscard]] Event get();

  [[nodiscard]] Event get(CUcontext context);

  void put(CUevent event, CUcontext context) noexcept;

  [[nodiscard]] std::size_t num_free_events(CUcontext context) const;

  [[nodiscard]] std::size_t total_free_events() const;

  static EventPool& instance();
};
}  // namespace kvikio::detail
