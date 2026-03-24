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
/**
 * @brief Thread-safe singleton pool for reusable CUDA events
 *
 * Manages a pool of CUDA events organized by CUDA context. Events are retained and reused across
 * calls to minimize allocation overhead. Each context maintains its own separate pool of events
 * since CUDA events are context-specific resources.
 *
 * All events are created with `CU_EVENT_DISABLE_TIMING` for minimal overhead.
 *
 * Call `EventPool::instance().get()` to acquire an event that will be automatically returned to the
 * pool when it goes out of scope (RAII).
 *
 * @note The destructor intentionally leaks events to avoid CUDA cleanup issues when static
 * destructors run after CUDA context destruction. @sa BounceBufferPool
 */
class EventPool {
 public:
  /**
   * @brief RAII wrapper for a pooled CUDA event
   *
   * Automatically returns the event to the pool when destroyed. Provides access to the underlying
   * CUevent handle and common event operations (record, synchronize).
   *
   * @note Non-copyable but movable to allow transfer of ownership while maintaining RAII
   */
  class Event {
    friend class EventPool;

   private:
    EventPool* _pool{};
    CUevent _event{};
    CUcontext _cuda_context{};

    /**
     * @brief Construct an Event wrapping a CUDA event handle
     *
     * @param pool The owning EventPool to return this event to on destruction
     * @param event The CUDA event handle to wrap
     * @param context The CUDA context associated with this event
     */
    explicit Event(EventPool* pool, CUevent event, CUcontext context) noexcept;

   public:
    ~Event() noexcept;

    // Move-only
    Event(Event const&)            = delete;
    Event& operator=(Event const&) = delete;
    Event(Event&& o) noexcept;
    Event& operator=(Event&& o) noexcept;

    /**
     * @brief Get the underlying CUDA event handle
     *
     * @return The CUevent handle wrapped by this object
     */
    [[nodiscard]] CUevent get() const noexcept;

    /**
     * @brief Get the CUDA context associated with this event
     *
     * @return The CUcontext this event belongs to
     */
    [[nodiscard]] CUcontext cuda_context() const noexcept;

    /**
     * @brief Record the event on a CUDA stream
     *
     * Records the event to capture the current state of the stream. The event will be signaled when
     * all preceding operations on the stream have completed.
     *
     * @param stream The CUDA stream to record the event on (must belong to the same context as this
     * event)
     *
     * @exception kvikio::CUfileException if the record operation fails
     */
    void record(CUstream stream);

    /**
     * @brief Block the calling thread until the event has been signaled
     *
     * Waits for all work captured by a preceding record() call to complete.
     *
     * @exception kvikio::CUfileException if the synchronize operation fails
     */
    void synchronize();
  };

 private:
  std::mutex mutable _mutex;
  // Per-context pools of free events
  std::unordered_map<CUcontext, std::vector<CUevent>> _pools;

  EventPool() = default;

  // Intentionally leak events during static destruction. @sa BounceBufferPool
  ~EventPool() noexcept = default;

 public:
  // Non-copyable, non-movable singleton
  EventPool(EventPool const&)            = delete;
  EventPool& operator=(EventPool const&) = delete;
  EventPool(EventPool&&)                 = delete;
  EventPool& operator=(EventPool&&)      = delete;

  /**
   * @brief Acquire a CUDA event from the pool for the current CUDA context.
   *
   * Returns a cached event for the current CUDA context if available, otherwise creates a new one.
   * The returned Event object will automatically return the event to the pool when it goes out of
   * scope.
   *
   * @return RAII Event object wrapping the acquired CUDA event
   * @exception kvikio::CUfileException if no CUDA context is current or event creation fails
   */
  [[nodiscard]] Event get();

  /**
   * @brief Return an event to the pool for reuse
   *
   * Typically called automatically by Event's destructor. Adds the event to the pool associated
   * with its context for future reuse.
   *
   * @param event The CUDA event handle to return
   * @param context The CUDA context associated with the event
   */
  void put(CUevent event, CUcontext context) noexcept;

  /**
   * @brief Get the number of free events for a specific context
   *
   * @param context The CUDA context to query
   * @return The number of events available for reuse in that context's pool
   */
  [[nodiscard]] std::size_t num_free_events(CUcontext context) const;

  /**
   * @brief Get the total number of free events across all contexts
   *
   * @return The total count of events available for reuse
   */
  [[nodiscard]] std::size_t total_free_events() const;

  /**
   * @brief Get the singleton instance of the event pool
   *
   * @return Reference to the singleton EventPool instance
   */
  static EventPool& instance();
};
}  // namespace kvikio::detail
