/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>

#include <kvikio/detail/event.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

/**
 * @brief Per-pread event barrier used to gate H2D completion on the caller's thread.
 *
 * One `IoEventBarrier` is constructed per `RemoteHandle::pread` call (device-buffer path) and
 * shared via `std::shared_ptr` with every sub-range transfer belonging to that pread. Each reactor
 * I/O thread that submits a `cuMemcpyAsync` for one of those sub-ranges calls
 * `record_event(stream)`, which re-records this thread's slot in the barrier. Once all sub-ranges
 * have reported completion, the caller calls `sync_all_events()` to block until every reactor
 * thread's last H2D has drained.
 *
 * The map is keyed by `std::this_thread::get_id()` so multiple reactor threads each get an
 * independent event slot. Subsequent records on the same thread overwrite the slot's captured state
 * via `cuEventRecord`.
 */
class IoEventBarrier {
 public:
  /**
   * @brief Construct a barrier carrying `cuda_context` as metadata.
   *
   * @param cuda_context The CUDA context that pred's H2Ds will land in. Stored only for callers
   * to look up. The barrier itself does not push or use this context.
   */
  explicit IoEventBarrier(CUcontext cuda_context) noexcept;

  IoEventBarrier(IoEventBarrier const&)            = delete;
  IoEventBarrier& operator=(IoEventBarrier const&) = delete;
  IoEventBarrier(IoEventBarrier&&)                 = delete;
  IoEventBarrier& operator=(IoEventBarrier&&)      = delete;

  ~IoEventBarrier() noexcept = default;

  /**
   * @brief Get the CUDA context this barrier was constructed with.
   *
   * @return The stored `CUcontext`.
   */
  [[nodiscard]] CUcontext cuda_context() const noexcept;

  /**
   * @brief Record an event on `stream` for the calling thread.
   *
   * Creates the calling thread's slot on first call. Subsequent calls on the same thread re-record
   * on the same event handle, overwriting any prior captured state.
   *
   * @param stream The CUDA stream to record on. Must belong to the same context as the event
   * (i.e. the context that was current at first-record).
   *
   * @exception kvikio::CUfileException if the underlying `cuEventRecord` fails.
   */
  void record_event(CUstream stream);

  /**
   * @brief Block the calling thread until every recorded event has signaled.
   *
   * Iterates each thread slot and calls `synchronize()`. After this returns, all H2Ds captured by
   * the last `record_event` call on each reactor thread have completed. Context-agnostic on the
   * calling thread.
   *
   * @exception kvikio::CUfileException if any underlying `cuEventSynchronize` fails.
   */
  void sync_all_events();

 private:
  CUcontext _cuda_context;
  std::mutex _mutex;
  std::unordered_map<std::thread::id, CudaEventPool::CudaEvent> _thread_events;
};

}  // namespace kvikio::detail
