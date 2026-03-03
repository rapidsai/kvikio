/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <map>
#include <mutex>
#include <thread>

#include <kvikio/detail/event.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {
class IoContext {
 public:
  CUcontext _cuda_context;
  std::mutex mutable _mutex;
  std::map<std::thread::id, EventPool::Event> _thread_events;

 public:
  explicit IoContext(CUcontext cuda_context) noexcept;

  ~IoContext() = default;

  // Non-copyable, non-movable (shared via shared_ptr)
  IoContext(IoContext const&)            = delete;
  IoContext& operator=(IoContext const&) = delete;
  IoContext(IoContext&&)                 = delete;
  IoContext& operator=(IoContext&&)      = delete;

  [[nodiscard]] CUcontext cuda_context() const noexcept;

  void record_event(CUstream stream);

  void sync_all_events();
};
}  // namespace kvikio::detail
