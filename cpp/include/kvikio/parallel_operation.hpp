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

#include <future>
#include <numeric>
#include <system_error>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <kvikio/error.hpp>
#include <kvikio/thread_pool/default.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief Apply read or write operation in parallel by submitting `ntasks` tasks.
 *
 * @tparam T The type of the function applying the read or write operation.
 * @param op The function applying the read or write operation.
 * @param devPtr Device pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param ntasks Number of tasks to submit.
 * @return A future to be used later to check if the operation has finished its execution.
 */
template <typename T>
std::future<std::size_t> parallel_io(
  T op, const void* devPtr, std::size_t size, std::size_t file_offset, std::size_t ntasks)
{
  auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(devPtr);

  int device{-1};
  if (cudaGetDevice(&device) != cudaSuccess) { throw CUfileException("cudaGetDevice failed"); }

  auto task = [device, op](void* devPtr_base,
                           std::size_t size,
                           std::size_t file_offset,
                           std::size_t devPtr_offset) -> std::size_t {
    if (device > -1) {
      if (cudaSetDevice(device) != cudaSuccess) { throw CUfileException("cudaSetDevice failed"); }
    }
    return op(devPtr_base, size, file_offset, devPtr_offset);
  };

  const std::size_t tasksize = size / ntasks;
  std::size_t last_jobsize   = tasksize + size % ntasks;
  std::vector<std::future<std::size_t>> tasks;
  tasks.reserve(ntasks);
  for (std::size_t i = 0; i < ntasks; ++i) {
    const std::size_t cur_size = (i == ntasks - 1) ? last_jobsize : tasksize;
    const std::size_t offset   = i * tasksize;
    tasks.push_back(default_thread_pool::get().submit(
      task, devPtr_base, cur_size, file_offset + offset, devPtr_offset + offset));
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
