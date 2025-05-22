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

#include <chrono>
#include <cstring>
#include <future>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

std::size_t get_page_size();

[[nodiscard]] off_t convert_size2off(std::size_t x);

[[nodiscard]] ssize_t convert_size2ssize(std::size_t x);

[[nodiscard]] CUdeviceptr convert_void2deviceptr(void const* devPtr);

/**
 * @brief Help function to convert value to 64 bit signed integer
 */
template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
[[nodiscard]] std::int64_t convert_to_64bit(T value)
{
  if constexpr (std::numeric_limits<T>::max() > std::numeric_limits<std::int64_t>::max()) {
    KVIKIO_EXPECT(value <= std::numeric_limits<std::int64_t>::max(),
                  "convert_to_64bit(x): x too large to fit std::int64_t",
                  std::overflow_error);
  }
  return std::int64_t(value);
}

/**
 * @brief Helper function to allow NVTX payload of type std::uint64_t to pass through without doing
 * anything.
 */
[[nodiscard]] inline std::uint64_t convert_to_64bit(std::uint64_t value) { return value; }

/**
 * @brief Help function to convert value to 64 bit float
 */
template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
[[nodiscard]] double convert_to_64bit(T value)
{
  return double(value);
}

/**
 * @brief Check if `ptr` points to host memory (as opposed to device memory)
 *
 * In this context, managed memory counts as device memory
 *
 * @param ptr Memory pointer to query
 * @return The boolean answer
 */
#ifdef KVIKIO_CUDA_FOUND
bool is_host_memory(void const* ptr);
#else
constexpr bool is_host_memory(void const* ptr) { return true; }
#endif

/**
 * @brief Return the device owning the pointer
 *
 * @param ptr Device pointer to query
 * @return The device ordinal
 */
[[nodiscard]] int get_device_ordinal_from_pointer(CUdeviceptr dev_ptr);

/**
 * @brief Given a device ordinal, return the primary context of the device.
 *
 * This function caches the primary contexts retrieved until program exit
 *
 * @param ordinal Device ordinal - an integer between 0 and the number of CUDA devices
 * @return Primary CUDA context
 */
[[nodiscard]] KVIKIO_EXPORT CUcontext get_primary_cuda_context(int ordinal);

/**
 * @brief Return the CUDA context associated the given device pointer, if any.
 *
 * @param dev_ptr Device pointer to query
 * @return Usable CUDA context, if one were found.
 */
[[nodiscard]] std::optional<CUcontext> get_context_associated_pointer(CUdeviceptr dev_ptr);

/**
 * @brief Check if the current CUDA context can access the given device pointer
 *
 * @param dev_ptr Device pointer to query
 * @return The boolean answer
 */
[[nodiscard]] bool current_context_can_access_pointer(CUdeviceptr dev_ptr);

/**
 * @brief Return a CUDA context that can be used with the given device pointer
 *
 * For robustness, we look for an usabale context in the following order:
 *   1) If a context has been associated with `devPtr`, it is returned.
 *   2) If the current context exists and can access `devPtr`, it is returned.
 *   3) Return the primary context of the device that owns `devPtr`. We assume the
 *      primary context can access `devPtr`, which might not be true in the exceptional
 *      disjoint addressing cases mention in the CUDA docs[1]. In these cases, the user
 *      has to set an usable current context before reading/writing using KvikIO.
 *
 * [1] <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html>
 *
 * @param devPtr Device pointer to query
 * @return Usable CUDA context
 */
[[nodiscard]] CUcontext get_context_from_pointer(void const* devPtr);

/**
 * @brief Push CUDA context on creation and pop it on destruction
 */
class PushAndPopContext {
 private:
  CUcontext _ctx;

 public:
  PushAndPopContext(CUcontext ctx);
  PushAndPopContext(PushAndPopContext const&)            = delete;
  PushAndPopContext& operator=(PushAndPopContext const&) = delete;
  PushAndPopContext(PushAndPopContext&&)                 = delete;
  PushAndPopContext&& operator=(PushAndPopContext&&)     = delete;
  ~PushAndPopContext();
};

// Find the base and offset of the memory allocation `devPtr` is in
std::tuple<void*, std::size_t, std::size_t> get_alloc_info(void const* devPtr,
                                                           CUcontext* ctx = nullptr);

/**
 * @brief Create a shared state in a future object that is immediately ready.
 *
 * A partial implementation of the namesake function from the concurrency TS
 * (https://en.cppreference.com/w/cpp/experimental/make_ready_future). The cases of
 * std::reference_wrapper and void are not implemented.
 *
 * @tparam T Type of the value provided.
 * @param t Object provided.
 * @return A future holding a decayed copy of the object provided.
 */
template <typename T>
std::future<std::decay_t<T>> make_ready_future(T&& t)
{
  std::promise<std::decay_t<T>> p;
  auto fut = p.get_future();
  p.set_value(std::forward<T>(t));
  return fut;
}

/**
 * @brief Check the status of the future object. True indicates that the result is available in the
 * future's shared state. False otherwise.
 *
 * The future shall not be created using `std::async(std::launch::deferred)`. Otherwise, this
 * function always returns true.
 *
 * @tparam T Type of the future.
 * @param future Instance of the future.
 * @return Boolean answer indicating if the future is ready or not.
 */
template <typename T>
bool is_future_done(T const& future)
{
  KVIKIO_EXPECT(future.valid(),
                "The future object does not refer to a valid shared state.",
                std::invalid_argument);
  return future.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
}

}  // namespace kvikio
