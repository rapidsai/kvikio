/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cassert>
#include <future>
#include <numeric>
#include <system_error>
#include <utility>
#include <vector>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief Apply read or write operation in parallel.
 *
 * @tparam T The type of the function applying the read or write operation.
 * @param op The function applying the read or write operation.
 * @param devPtr Device pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param task_size Size of each task in bytes.
 * @return A future to be used later to check if the operation has finished its execution.
 */
template <typename T>
std::future<std::size_t> parallel_io(
  T op, const void* devPtr, std::size_t size, std::size_t file_offset, std::size_t task_size)
{
  if (task_size == 0) { throw std::invalid_argument("`task_size` cannot be zero"); }

  CUcontext ctx                                = get_context_from_device_pointer(devPtr);
  auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(devPtr, &ctx);

  auto task = [op, ctx](void* devPtr_base,
                        std::size_t size,
                        std::size_t file_offset,
                        std::size_t devPtr_offset) -> std::size_t {
    PushAndPopContext c(ctx);
    return op(devPtr_base, size, file_offset, devPtr_offset);
  };

  // Single-task guard
  if (task_size >= size || page_size >= size) {
    return defaults::thread_pool().submit(task, devPtr_base, size, file_offset, devPtr_offset);
  }

  // We know an upper bound of the total number of tasks
  std::vector<std::future<std::size_t>> tasks;
  tasks.reserve(size / task_size + 2);

  // 1) Submit tasks for the aligned range from the first page boundary to the last page boundary
  {
    while (size >= task_size) {
      tasks.push_back(
        defaults::thread_pool().submit(task, devPtr_base, task_size, file_offset, devPtr_offset));
      file_offset += task_size;
      devPtr_offset += task_size;
      size -= task_size;
    }
  }
  // 2) Submit a task for the remainder range from the last page boundary to the end
  if (size > 0) {
    tasks.push_back(
      defaults::thread_pool().submit(task, devPtr_base, size, file_offset, devPtr_offset));
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
