/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <atomic>
#include <cassert>
#include <future>
#include <mutex>
#include <numeric>
#include <system_error>
#include <utility>
#include <vector>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

namespace detail {

/**
 * @brief Determine the NVTX color and call index. They are used to identify tasks from different
 * pread/pwrite calls. Tasks from the same pread/pwrite call are given the same color and call
 * index. The call index is atomically incremented on each pread/pwrite call, and will wrap around
 * once it reaches the maximum value the integer type `std::uint64_t` can hold (this overflow
 * behavior is well-defined in C++). The color is picked from an internal color palette according to
 * the call index value.
 *
 * @return A pair of NVTX color and call index.
 */
inline const std::pair<const nvtx_color_type&, std::uint64_t> get_next_color_and_call_idx() noexcept
{
  static std::atomic_uint64_t call_counter{1ull};
  auto call_idx    = call_counter.fetch_add(1ull, std::memory_order_relaxed);
  auto& nvtx_color = nvtx_manager::get_color_by_index(call_idx);
  return {nvtx_color, call_idx};
}

template <typename F, typename T>
std::future<std::size_t> submit_task(F op,
                                     T buf,
                                     std::size_t size,
                                     std::size_t file_offset,
                                     std::size_t devPtr_offset,
                                     std::uint64_t nvtx_payload = 0ull,
                                     nvtx_color_type nvtx_color = nvtx_manager::default_color())
{
  return defaults::thread_pool().submit_task([=] {
    KVIKIO_NVTX_SCOPED_RANGE("task", nvtx_payload, nvtx_color);

    // Rename the worker thread in the thread pool to improve clarity from nsys-ui.
    // Note: This NVTX feature is currently not supported by nsys-ui.
    thread_local std::once_flag call_once_per_thread;
    std::call_once(call_once_per_thread,
                   [] { nvtx_manager::rename_current_thread("thread pool"); });

    return op(buf, size, file_offset, devPtr_offset);
  });
}

}  // namespace detail

/**
 * @brief Apply read or write operation in parallel.
 *
 * @tparam F The type of the function applying the read or write operation.
 * @tparam T The type of the memory pointer.
 * @param op The function applying the read or write operation.
 * @param buf Buffer pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param task_size Size of each task in bytes.
 * @return A future to be used later to check if the operation has finished its execution.
 */
template <typename F, typename T>
std::future<std::size_t> parallel_io(F op,
                                     T buf,
                                     std::size_t size,
                                     std::size_t file_offset,
                                     std::size_t task_size,
                                     std::size_t devPtr_offset,
                                     std::uint64_t call_idx     = 0,
                                     nvtx_color_type nvtx_color = nvtx_manager::default_color())
{
  if (task_size == 0) { throw std::invalid_argument("`task_size` cannot be zero"); }

  // Single-task guard
  if (task_size >= size || page_size >= size) {
    return detail::submit_task(op, buf, size, file_offset, devPtr_offset, call_idx, nvtx_color);
  }

  // We know an upper bound of the total number of tasks
  std::vector<std::future<std::size_t>> tasks;
  tasks.reserve(size / task_size + 2);

  // 1) Submit `task_size` sized tasks
  while (size >= task_size) {
    tasks.push_back(
      detail::submit_task(op, buf, task_size, file_offset, devPtr_offset, call_idx, nvtx_color));
    file_offset += task_size;
    devPtr_offset += task_size;
    size -= task_size;
  }

  // 2) Submit a task for the remainder
  if (size > 0) {
    tasks.push_back(
      detail::submit_task(op, buf, size, file_offset, devPtr_offset, call_idx, nvtx_color));
  }

  // Finally, we sum the result of all tasks.
  auto gather_tasks = [](std::vector<std::future<std::size_t>>&& tasks) -> std::size_t {
    std::size_t ret = 0;
    for (auto& task : tasks) {
      ret += task.get();
    }
    return ret;
  };
  return std::async(std::launch::deferred, gather_tasks, std::move(tasks));
}

}  // namespace kvikio
