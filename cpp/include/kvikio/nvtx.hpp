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

using nvtx_scoped_range_type      = nvtx3::scoped_range_in<libkvikio_domain>;
using nvtx_registered_string_type = nvtx3::registered_string_in<libkvikio_domain>;

// Macro to concatenate two tokens x and y.
#define KVIKIO_CONCAT_HELPER(x, y) x##y
#define KVIKIO_CONCAT(x, y)        KVIKIO_CONCAT_HELPER(x, y)

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define KVIKIO_REGISTER_STRING(message)                              \
  [](const char* a_message) -> auto& {                               \
    static kvikio::nvtx_registered_string_type a_reg_str{a_message}; \
    return a_reg_str;                                                \
  }(message)

// Implementation of KVIKIO_NVTX_FUNC_RANGE()
// todo: Although supported by many compilers, __PRETTY_FUNCTION__ is non-standard. Replacement may
// be considered once reflection is standardized.
#define KVIKIO_NVTX_FUNC_RANGE_IMPL_0() KVIKIO_NVTX_SCOPED_RANGE_IMPL_1(__PRETTY_FUNCTION__)
#define KVIKIO_NVTX_FUNC_RANGE_IMPL_1(payload) \
  KVIKIO_NVTX_SCOPED_RANGE_IMPL_2(__PRETTY_FUNCTION__, payload)
#define KVIKIO_NVTX_FUNC_RANGE_IMPL_2(payload, color) \
  KVIKIO_NVTX_SCOPED_RANGE_IMPL_3(__PRETTY_FUNCTION__, payload, color)
#define KVIKIO_NVTX_FUNC_RANGE_SELECTOR(_0, _1, _2, NAME, ...) NAME
// todo: Although supported by gcc and clang, ##__VA_ARGS__ is non-standard, and should be replaced
// by __VA_OPT__ (since C++20) in the future.
#define KVIKIO_NVTX_FUNC_RANGE_IMPL(...)                         \
  KVIKIO_NVTX_FUNC_RANGE_SELECTOR(_0,                            \
                                  ##__VA_ARGS__,                 \
                                  KVIKIO_NVTX_FUNC_RANGE_IMPL_2, \
                                  KVIKIO_NVTX_FUNC_RANGE_IMPL_1, \
                                  KVIKIO_NVTX_FUNC_RANGE_IMPL_0) \
  (__VA_ARGS__)

// Implementation of KVIKIO_NVTX_SCOPED_RANGE(...)
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_1(message)                             \
  kvikio::nvtx_scoped_range_type KVIKIO_CONCAT(_kvikio_nvtx_range, __LINE__) \
  {                                                                          \
    nvtx3::event_attributes                                                  \
    {                                                                        \
      KVIKIO_REGISTER_STRING(message), kvikio::NvtxManager::default_color()  \
    }                                                                        \
  }
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_3(message, payload_v, color)                                \
  kvikio::nvtx_scoped_range_type KVIKIO_CONCAT(_kvikio_nvtx_range, __LINE__)                      \
  {                                                                                               \
    nvtx3::event_attributes                                                                       \
    {                                                                                             \
      KVIKIO_REGISTER_STRING(message), nvtx3::payload{kvikio::convert_to_64bit(payload_v)}, color \
    }                                                                                             \
  }
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_2(message, payload) \
  KVIKIO_NVTX_SCOPED_RANGE_IMPL_3(message, payload, kvikio::NvtxManager::default_color())
#define KVIKIO_NVTX_SCOPED_RANGE_SELECTOR(_1, _2, _3, NAME, ...) NAME
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL(...)                           \
  KVIKIO_NVTX_SCOPED_RANGE_SELECTOR(__VA_ARGS__,                     \
                                    KVIKIO_NVTX_SCOPED_RANGE_IMPL_3, \
                                    KVIKIO_NVTX_SCOPED_RANGE_IMPL_2, \
                                    KVIKIO_NVTX_SCOPED_RANGE_IMPL_1) \
  (__VA_ARGS__)

// Implementation of KVIKIO_NVTX_MARKER(message, payload)
#define KVIKIO_NVTX_MARKER_IMPL(message, payload_v)                 \
  nvtx3::mark_in<kvikio::libkvikio_domain>(nvtx3::event_attributes{ \
    KVIKIO_REGISTER_STRING(message), nvtx3::payload{kvikio::convert_to_64bit(payload_v)}})

#endif

#ifdef KVIKIO_CUDA_FOUND
using nvtx_color_type = nvtx3::color;
#else
using nvtx_color_type = int;
#endif

/**
 * @brief Utility singleton class for NVTX annotation.
 */
class NvtxManager {
 public:
  static NvtxManager& instance() noexcept;

  /**
   * @brief Return the default color.
   *
   * @return Default color.
   */
  static const nvtx_color_type& default_color() noexcept;

  /**
   * @brief Return the color at the given index from the internal color palette whose size n is a
   * power of 2. The index may exceed the size of the color palette, in which case it wraps around,
   * i.e. (idx mod n).
   *
   * @param idx The index value.
   * @return The color picked from the internal color palette.
   */
  static const nvtx_color_type& get_color_by_index(std::uint64_t idx) noexcept;

  /**
   * @brief Rename the current thread under the KvikIO NVTX domain.
   *
   * @note This NVTX feature is currently not supported by the Nsight System profiler. As a result,
   * the OS thread will not be renamed in the nsys-ui.
   */
  static void rename_current_thread(std::string_view new_name) noexcept;

  NvtxManager(NvtxManager const&)            = delete;
  NvtxManager& operator=(NvtxManager const&) = delete;
  NvtxManager(NvtxManager&&)                 = delete;
  NvtxManager& operator=(NvtxManager&&)      = delete;

 private:
  NvtxManager() = default;
};

/**
 * @brief Convenience macro for generating an NVTX range in the `libkvikio` domain from the lifetime
 * of a function. Can be used inside a regular function or a lambda expression.
 *
 * The function name contains detailed information such as namespace, return type, parameter type,
 * etc.
 *
 * @param payload (Optional) NVTX payload.
 * @param color (Optional) NVTX color. If unspecified, a default NVTX color is used.
 *
 * Example:
 * ```
 * void some_function(){
 *    // No argument
 *    KVIKIO_NVTX_FUNC_RANGE();
 *    ...
 * }
 *
 * void some_function(){
 *    // Specify payload
 *    KVIKIO_NVTX_FUNC_RANGE(4096);
 *    ...
 * }
 *
 * void some_function(){
 *    // Specify payload and color
 *    auto const nvtx3::rgb color{0, 255, 0};
 *    KVIKIO_NVTX_FUNC_RANGE(4096, color);
 *    ...
 * }
 * ```
 */
#ifdef KVIKIO_CUDA_FOUND
#define KVIKIO_NVTX_FUNC_RANGE(...) KVIKIO_NVTX_FUNC_RANGE_IMPL(__VA_ARGS__)
#else
#define KVIKIO_NVTX_FUNC_RANGE(...) \
  do {                              \
  } while (0)
#endif

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `libkvikio` domain to
 * annotate a time duration.
 *
 * @param message String literal for NVTX annotation. To improve profile-time performance, the
 * string literal is registered in NVTX.
 * @param payload (Optional) NVTX payload.
 * @param color (Optional) NVTX color. If unspecified, a default NVTX color is used.
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
#define KVIKIO_NVTX_SCOPED_RANGE(message, payload, ...) \
  do {                                                  \
  } while (0)
#endif

/**
 * @brief Convenience macro for generating an NVTX marker in the `libkvikio` domain to annotate a
 * certain time point.
 *
 * @param message String literal for NVTX annotation. To improve profile-time performance, the
 * string literal is registered in NVTX.
 * @param payload NVTX payload.
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
