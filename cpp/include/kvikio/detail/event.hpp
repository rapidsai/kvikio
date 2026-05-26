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
 * Call `CudaEventPool::instance().get()` to acquire an event that is bound to the CUDA context
 * currently set on the calling thread. The event will be automatically returned to the pool when it
 * goes out of scope (RAII).
 *
 * @note The destructor intentionally does NOT call `cuEventDestroy` on cached events.
 * `CudaEventPool::instance()` is a function-local static destructed after `main` returns, and
 * making CUDA driver API calls from a static object's destructor at that point is undefined
 * behavior per the CUDA programming guide:
 * https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#runtime-initialization
 * The OS reclaims process memory at exit regardless.
 */
class CudaEventPool {
 public:
  /**
   * @brief RAII wrapper for a pooled CUDA event
   *
   * Automatically returns the event to the pool when destroyed. Provides access to the underlying
   * CUevent handle and common event operations (record, synchronize).
   *
   * @note Non-copyable but movable to allow transfer of ownership while maintaining RAII
   */
  class CudaEvent {
    friend class CudaEventPool;

   private:
    CudaEventPool* _pool{};
    CUevent _event{};
    CUcontext _cuda_context{};

    /**
     * @brief Construct a CudaEvent wrapping a CUDA event handle
     *
     * @param pool The owning CudaEventPool to return this event to on destruction
     * @param event The CUDA event handle to wrap
     * @param context The CUDA context associated with this event
     */
    explicit CudaEvent(CudaEventPool* pool, CUevent event, CUcontext context) noexcept;

   public:
    ~CudaEvent() noexcept;

    // Move-only
    CudaEvent(CudaEvent const&)            = delete;
    CudaEvent& operator=(CudaEvent const&) = delete;
    CudaEvent(CudaEvent&& o) noexcept;
    CudaEvent& operator=(CudaEvent&& o) noexcept;

    /**
     * @brief Get the underlying CUDA event handle
     *
     * @return The CUevent handle wrapped by this object
     */
    [[nodiscard]] CUevent get() const noexcept;

    /**
     * @brief Get the CUDA context associated with this event
     *
     * @return The CUcontext this event belongs to. Returns nullptr for a moved-from CudaEvent.
     */
    [[nodiscard]] CUcontext cuda_context() const noexcept;

    /**
     * @brief Record the event on a CUDA stream
     *
     * Records the event to capture the current state of the stream. The event will be signaled when
     * all preceding operations on the stream have completed.
     *
     * @param stream The CUDA stream to record the event on. Must belong to the same context as this
     * event. Otherwise CUDA returns an error.
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

    /**
     * @brief Non-blocking check whether all work captured by the event has completed.
     *
     * Returns true if all work captured by a preceding `record()` call has completed, false if work
     * is still pending. This is the non-blocking counterpart to `synchronize()`.
     *
     * @note An event that has never been recorded by `cudaAPI::instance().EventRecord()` reports
     * `is_done() == true`, since CUDA's `cuEventQuery` returns `CUDA_SUCCESS` when there is no
     * captured work. Users that rely on `is_done() == true` as a clean baseline signal should
     * `synchronize()` before dropping the event so that the next pool acquirer
     * `CudaEventPool::instance().get()` sees an idle state.
     *
     * @return true if the event has completed (or has never been recorded), false if work is still
     * in progress.
     *
     * @exception kvikio::CUfileException if the underlying `cuEventQuery` returns an error other
     * than `CUDA_SUCCESS` or `CUDA_ERROR_NOT_READY`.
     */
    [[nodiscard]] bool is_done() const;
  };

 private:
  std::mutex mutable _mutex;
  // Per-context pools of free events
  std::unordered_map<CUcontext, std::vector<CUevent>> _pools;

  CudaEventPool() = default;

  // Intentionally `noexcept = default`. See the class-level @note above: issuing CUDA driver
  // API calls (e.g., cuEventDestroy) from this destructor would be UB because the singleton is
  // destructed after main returns. The defaulted destructor runs ~_pools, which tears down the
  // std::vector<CUevent> entries without touching the handles.
  ~CudaEventPool() noexcept = default;

  /**
   * @brief Return an event to the pool for reuse
   *
   * Called by CudaEvent's destructor (and move-assignment operator) via the friend declaration.
   * Adds the event to the pool associated with its context for future reuse.
   *
   * @param event The CUDA event handle to return
   * @param context The CUDA context associated with the event
   *
   * @note noexcept: any failure inside push_back (e.g., allocator failure) is caught and logged.
   * The event is then destroyed instead of being cached.
   */
  void put(CUevent event, CUcontext context) noexcept;

 public:
  // Non-copyable, non-movable singleton
  CudaEventPool(CudaEventPool const&)            = delete;
  CudaEventPool& operator=(CudaEventPool const&) = delete;
  CudaEventPool(CudaEventPool&&)                 = delete;
  CudaEventPool& operator=(CudaEventPool&&)      = delete;

  /**
   * @brief Acquire a CUDA event for the CUDA context currently set on the calling thread.
   *
   * Returns a cached event for the current CUDA context if available, otherwise creates a new one.
   * The returned CudaEvent object will automatically return the event to the pool when it goes out
   * of scope.
   *
   * @return RAII CudaEvent object wrapping the acquired CUDA event
   * @exception kvikio::CUfileException if no CUDA context is current or event creation fails
   */
  [[nodiscard]] CudaEvent get();

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
   * @return Reference to the singleton CudaEventPool instance
   */
  static CudaEventPool& instance();
};
}  // namespace kvikio::detail
