/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cstdint>

#ifdef KVIKIO_CUDA_FOUND
#include <nvtx3/nvtx3.hpp>
#endif

#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

#ifdef KVIKIO_CUDA_FOUND
/**
 * @brief Tag type for libkvikio's NVTX domain.
 */
struct libkvikio_domain {
  static constexpr char const* name{"libkvikio"};
};

using nvtx_scoped_range      = nvtx3::scoped_range_in<libkvikio_domain>;
using nvtx_registered_string = nvtx3::registered_string_in<libkvikio_domain>;

// Macro to concatenate two tokens x and y.
#define KVIKIO_CONCAT_HELPER(x, y) x##y
#define KVIKIO_CONCAT(x, y)        KVIKIO_CONCAT_HELPER(x, y)

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define KVIKIO_REGISTER_STRING(msg)                 \
  [](const char* a_msg) -> auto& {                  \
    static nvtx_registered_string a_reg_str{a_msg}; \
    return a_reg_str;                               \
  }(msg)

// Macro overloads of KVIKIO_NVTX_FUNC_RANGE
#define KVIKIO_NVTX_FUNC_RANGE_IMPL() NVTX3_FUNC_RANGE_IN(libkvikio_domain)

#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_3(msg, nvtx_payload_v, nvtx_color_v)                        \
  kvikio::nvtx_scoped_range KVIKIO_CONCAT(_kvikio_nvtx_range, __LINE__)                           \
  {                                                                                               \
    nvtx3::event_attributes                                                                       \
    {                                                                                             \
      KVIKIO_REGISTER_STRING(msg), nvtx3::payload{convert_to_64bit(nvtx_payload_v)}, nvtx_color_v \
    }                                                                                             \
  }
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_2(msg, nvtx_payload_v) \
  KVIKIO_NVTX_SCOPED_RANGE_IMPL_3(                           \
    msg, nvtx_payload_v, kvikio::nvtx_manager::instance().default_color())
#define KVIKIO_NVTX_SCOPED_RANGE_SELECTOR(_1, _2, _3, NAME, ...) NAME
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL(...)                                         \
  KVIKIO_NVTX_SCOPED_RANGE_SELECTOR(                                               \
    __VA_ARGS__, KVIKIO_NVTX_SCOPED_RANGE_IMPL_3, KVIKIO_NVTX_SCOPED_RANGE_IMPL_2) \
  (__VA_ARGS__)

#define KVIKIO_NVTX_MARKER_IMPL(msg, nvtx_payload_v)        \
  nvtx3::mark_in<libkvikio_domain>(nvtx3::event_attributes{ \
    KVIKIO_REGISTER_STRING(msg), nvtx3::payload{convert_to_64bit(nvtx_payload_v)}})

#endif

#ifdef KVIKIO_CUDA_FOUND
using nvtx_color = nvtx3::color;
#else
using nvtx_color = int;
#endif

class nvtx_manager {
 public:
  static nvtx_manager& instance() noexcept;
  const nvtx_color& default_color() const noexcept;
  const nvtx_color& get_color_by_index(std::uint64_t idx) const noexcept;
  nvtx_manager(nvtx_manager const&)            = delete;
  nvtx_manager& operator=(nvtx_manager const&) = delete;
  nvtx_manager(nvtx_manager&&)                 = delete;
  nvtx_manager& operator=(nvtx_manager&&)      = delete;

 private:
  nvtx_manager() = default;
};

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
 * Takes two arguments (message, nvtx_payload_v).
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
#define KVIKIO_NVTX_SCOPED_RANGE(...) KVIKIO_NVTX_SCOPED_RANGE_IMPL(__VA_ARGS__)
#else
#define KVIKIO_NVTX_SCOPED_RANGE(msg, nvtx_payload_v, ...) \
  do {                                                     \
  } while (0)
#endif

/**
 * @brief Convenience macro for generating an NVTX marker in the `libkvikio` domain to annotate a
 * certain time point.
 *
 * Takes two arguments (message, nvtx_payload_v). Use this macro to annotate asynchronous I/O
 * operations, where the nvtx_payload_v refers to the I/O size.
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
#define KVIKIO_NVTX_MARKER(message, nvtx_payload_v) KVIKIO_NVTX_MARKER_IMPL(message, nvtx_payload_v)
#else
#define KVIKIO_NVTX_MARKER(message, nvtx_payload_v) \
  do {                                              \
  } while (0)
#endif

}  // namespace kvikio
