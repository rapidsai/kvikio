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
#include <memory>
#include <numeric>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

namespace detail {

/**
 * @brief Utility function to create a copyable callable from a move-only callable.
 *
 * The underlying thread pool uses `std::function` (until C++23) or `std::move_only_function`
 * (since C++23) as the element type of the task queue. For the former case that currently applies,
 * the `std::function` requires its "target" (associated callable) to be copy-constructible. This
 * utility function is a workaround for those move-only callables.
 *
 * @tparam F Callable type. F shall be move-only.
 * @param op Callable.
 * @return A new callable that satisfies the copy-constructible condition.
 */
template <typename F>
auto make_copyable_lambda(F op)
{
  // Create the callable on the heap by moving from op. Use a shared pointer to manage its lifetime.
  auto sp = std::make_shared<F>(std::move(op));

  // Use the copyable closure as the proxy of the move-only callable.
  return
    [sp](auto&&... args) -> decltype(auto) { return (*sp)(std::forward<decltype(args)>(args)...); };
}

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
  auto& nvtx_color = NvtxManager::get_color_by_index(call_idx);
  return {nvtx_color, call_idx};
}

/**
 * @brief Submit the task callable to the underlying thread pool.
 *
 * Both the callable and arguments shall satisfy copy-constructible.
 */
template <typename F, typename T>
std::future<std::size_t> submit_task(F op,
                                     T buf,
                                     std::size_t size,
                                     std::size_t file_offset,
                                     std::size_t devPtr_offset,
                                     std::uint64_t nvtx_payload = 0ull,
                                     nvtx_color_type nvtx_color = NvtxManager::default_color())
{
  static_assert(std::is_invocable_r_v<std::size_t,
                                      decltype(op),
                                      decltype(buf),
                                      decltype(size),
                                      decltype(file_offset),
                                      decltype(devPtr_offset)>);

  return defaults::thread_pool().submit_task([=] {
    KVIKIO_NVTX_SCOPED_RANGE("task", nvtx_payload, nvtx_color);
    return op(buf, size, file_offset, devPtr_offset);
  });
}

/**
 * @brief Submit the move-only task callable to the underlying thread pool.
 *
 * @tparam F Callable type. F shall be move-only and have no argument.
 * @param op Callable.
 * @return A future to be used later to check if the operation has finished its execution.
 */
template <typename F>
std::future<std::size_t> submit_move_only_task(
  F op_move_only,
  std::uint64_t nvtx_payload = 0ull,
  nvtx_color_type nvtx_color = NvtxManager::default_color())
{
  static_assert(std::is_invocable_r_v<std::size_t, F>);
  auto op_copyable = make_copyable_lambda(std::move(op_move_only));
  return defaults::thread_pool().submit_task([=] {
    KVIKIO_NVTX_SCOPED_RANGE("task", nvtx_payload, nvtx_color);
    return op_copyable();
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
                                     nvtx_color_type nvtx_color = NvtxManager::default_color())
{
  KVIKIO_EXPECT(task_size > 0, "`task_size` must be positive", std::invalid_argument);
  static_assert(std::is_invocable_r_v<std::size_t,
                                      decltype(op),
                                      decltype(buf),
                                      decltype(size),
                                      decltype(file_offset),
                                      decltype(devPtr_offset)>);

  // Single-task guard
  if (task_size >= size || get_page_size() >= size) {
    return detail::submit_task(op, buf, size, file_offset, devPtr_offset, call_idx, nvtx_color);
  }

  std::vector<std::future<std::size_t>> tasks;
  tasks.reserve(size / task_size);

  // 1) Submit all tasks but the last one. These are all `task_size` sized tasks.
  while (size > task_size) {
    tasks.push_back(
      detail::submit_task(op, buf, task_size, file_offset, devPtr_offset, call_idx, nvtx_color));
    file_offset += task_size;
    devPtr_offset += task_size;
    size -= task_size;
  }

  // 2) Submit the last task, which consists of performing the last I/O and waiting the previous
  // tasks.
  auto last_task = [=, tasks = std::move(tasks)]() mutable -> std::size_t {
    auto ret = op(buf, size, file_offset, devPtr_offset);
    for (auto& task : tasks) {
      ret += task.get();
    }
    return ret;
  };
  return detail::submit_move_only_task(std::move(last_task), call_idx, nvtx_color);
}

}  // namespace kvikio
