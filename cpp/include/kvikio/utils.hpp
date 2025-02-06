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

#include <cassert>
#include <chrono>
#include <cstring>
#include <future>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#ifdef KVIKIO_CUDA_FOUND
#include <nvtx3/nvtx3.hpp>
#endif

#include <kvikio/shim/cuda.hpp>

namespace kvikio {

// cuFile defines a page size to 4 KiB
inline constexpr std::size_t page_size = 4096;

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
    if (value > std::numeric_limits<std::int64_t>::max()) {
      throw std::overflow_error("convert_to_64bit(x): x too large to fit std::int64_t");
    }
  }
  return std::int64_t(value);
}

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
  if (!future.valid()) {
    throw std::invalid_argument("The future object does not refer to a valid shared state.");
  }
  return future.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
}

#ifdef KVIKIO_CUDA_FOUND
/**
 * @brief Tag type for libkvikio's NVTX domain.
 */
struct libkvikio_domain {
  static constexpr char const* name{"libkvikio"};
};

// Macro to concatenate two tokens x and y.
#define KVIKIO_CONCAT_HELPER(x, y) x##y
#define KVIKIO_CONCAT(x, y)        KVIKIO_CONCAT_HELPER(x, y)

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define KVIKIO_REGISTER_STRING(msg)                                        \
  [](const char* a_msg) -> auto& {                                         \
    static nvtx3::registered_string_in<libkvikio_domain> a_reg_str{a_msg}; \
    return a_reg_str;                                                      \
  }(msg)

// Macro overloads of KVIKIO_NVTX_FUNC_RANGE
#define KVIKIO_NVTX_FUNC_RANGE_IMPL() NVTX3_FUNC_RANGE_IN(libkvikio_domain)

#define KVIKIO_NVTX_SCOPED_RANGE_IMPL(msg, val)                                        \
  nvtx3::scoped_range_in<libkvikio_domain> KVIKIO_CONCAT(_kvikio_nvtx_range, __LINE__) \
  {                                                                                    \
    nvtx3::event_attributes                                                            \
    {                                                                                  \
      KVIKIO_REGISTER_STRING(msg), nvtx3::payload { convert_to_64bit(val) }            \
    }                                                                                  \
  }

#define KVIKIO_NVTX_MARKER_IMPL(msg, val) \
  nvtx3::mark_in<libkvikio_domain>(       \
    nvtx3::event_attributes{KVIKIO_REGISTER_STRING(msg), nvtx3::payload{convert_to_64bit(val)}})

#endif

/**
 * @brief Convenience macro for generating an NVTX range in the `libkvikio` domain
 * from the lifetime of a function.
 *
 * Takes no argument. The name of the immediately enclosing function returned by `__func__` is used
 * as the message.
 *
 * Example:
 * ```
 * void some_function(){
 *    KVIKIO_NVTX_FUNC_RANGE();  // The name `some_function` is used as the message
 *    ...
 * }
 * ```
 */
#ifdef KVIKIO_CUDA_FOUND
#define KVIKIO_NVTX_FUNC_RANGE() KVIKIO_NVTX_FUNC_RANGE_IMPL()
#else
#define KVIKIO_NVTX_FUNC_RANGE(...) \
  do {                              \
  } while (0)
#endif

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `libkvikio` domain to
 * annotate a time duration.
 *
 * Takes two arguments (message, payload).
 *
 * Example:
 * ```
 * void some_function(){
 *    KVIKIO_NVTX_SCOPED_RANGE("my function", 42);
 *    ...
 * }
 * ```
 */
#ifdef KVIKIO_CUDA_FOUND
#define KVIKIO_NVTX_SCOPED_RANGE(msg, val) KVIKIO_NVTX_SCOPED_RANGE_IMPL(msg, val)
#else
#define KVIKIO_NVTX_SCOPED_RANGE(msg, val) \
  do {                                     \
  } while (0)
#endif

/**
 * @brief Convenience macro for generating an NVTX marker in the `libkvikio` domain to annotate a
 * certain time point.
 *
 * Takes two arguments (message, payload). Use this macro to annotate asynchronous I/O operations,
 * where the payload refers to the I/O size.
 *
 * Example:
 * ```
 * std::future<void> some_function(){
 *     size_t io_size{2077};
 *     KVIKIO_NVTX_MARKER("I/O operation", io_size);
 *     perform_async_io_operation(io_size);
 *     ...
 * }
 * ```
 */
#ifdef KVIKIO_CUDA_FOUND
#define KVIKIO_NVTX_MARKER(message, payload) KVIKIO_NVTX_MARKER_IMPL(message, payload)
#else
#define KVIKIO_NVTX_MARKER(message, payload) \
  do {                                       \
  } while (0)
#endif

}  // namespace kvikio
