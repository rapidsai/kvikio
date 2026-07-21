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
 * @brief Per-pread barrier that lets the caller's thread wait for every reactor's H2D to finish.
 *
 * Constructed once per device-path `RemoteHandle::pread` and shared via `std::shared_ptr` with all
 * of that pread's sub-range transfers. Each reactor I/O thread records into its own slot (keyed by
 * thread id) after every `cuMemcpyAsync`, re-recording the same event on later calls. Once all
 * sub-ranges report completion, the caller calls `sync_all_events()` to block until each thread's
 * last H2D has drained.
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
   * @brief Record an event on `stream` in the calling thread's slot, creating the slot on first
   * use.
   *
   * @param stream The CUDA stream to record on. Must belong to the same context as the event, i.e.
   * the context current at first record.
   * @exception kvikio::CUfileException if the underlying `cuEventRecord` fails.
   */
  void record_event(CUstream stream);

  /**
   * @brief Block the calling thread until every recorded event has signaled.
   *
   * After this returns, the last H2D recorded on each reactor thread has completed.
   * Context-agnostic on the calling thread.
   * @exception kvikio::CUfileException if any underlying `cuEventSynchronize` fails.
   */
  void sync_all_events();

 private:
  CUcontext _cuda_context;
  std::mutex _mutex;
  std::unordered_map<std::thread::id, CudaEventPool::CudaEvent> _thread_events;
};

}  // namespace kvikio::detail
