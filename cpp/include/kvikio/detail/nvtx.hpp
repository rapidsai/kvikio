/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include <nvtx3/nvtx3.hpp>

#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief Tag type for libkvikio's NVTX domain.
 */
struct libkvikio_domain {
  static constexpr char const* name{"libkvikio"};
};

using NvtxScopedRange      = nvtx3::scoped_range_in<libkvikio_domain>;
using NvtxRegisteredString = nvtx3::registered_string_in<libkvikio_domain>;
using NvtxColor            = nvtx3::color;

// Macro to concatenate two tokens x and y.
#define KVIKIO_CONCAT_HELPER(x, y) x##y
#define KVIKIO_CONCAT(x, y)        KVIKIO_CONCAT_HELPER(x, y)

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define KVIKIO_REGISTER_STRING(message)                       \
  [](const char* a_message) -> auto& {                        \
    static kvikio::NvtxRegisteredString a_reg_str{a_message}; \
    return a_reg_str;                                         \
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
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_1(message)                            \
  kvikio::NvtxScopedRange KVIKIO_CONCAT(_kvikio_nvtx_range, __LINE__)       \
  {                                                                         \
    nvtx3::event_attributes                                                 \
    {                                                                       \
      KVIKIO_REGISTER_STRING(message), kvikio::NvtxManager::default_color() \
    }                                                                       \
  }
#define KVIKIO_NVTX_SCOPED_RANGE_IMPL_3(message, payload_v, color)                                \
  kvikio::NvtxScopedRange KVIKIO_CONCAT(_kvikio_nvtx_range, __LINE__)                             \
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

struct NvtxContext {
  NvtxContext();
  NvtxContext(char const* file_name,
              std::size_t file_offse,
              std::size_t size,
              std::uint64_t call_idx,
              NvtxColor color);

  char const* file_name{};
  std::size_t file_offset{};
  std::size_t size{};
  std::uint64_t call_idx{};
  NvtxColor color;
};

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
  static const NvtxColor& default_color() noexcept;

  /**
   * @brief Return the color at the given index from the internal color palette whose size n is a
   * power of 2. The index may exceed the size of the color palette, in which case it wraps around,
   * i.e. (idx mod n).
   *
   * @param idx The index value.
   * @return The color picked from the internal color palette.
   */
  static const NvtxColor& get_color_by_index(std::uint64_t idx) noexcept;

  static NvtxContext get_next_call_context(char const* file_name,
                                           std::size_t file_offset,
                                           std::size_t size);

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
 *    nvtx3::rgb const color{0, 255, 0};
 *    KVIKIO_NVTX_FUNC_RANGE(4096, color);
 *    ...
 * }
 * ```
 */
#define KVIKIO_NVTX_FUNC_RANGE(...) KVIKIO_NVTX_FUNC_RANGE_IMPL(__VA_ARGS__)

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
#define KVIKIO_NVTX_SCOPED_RANGE(...) KVIKIO_NVTX_SCOPED_RANGE_IMPL(__VA_ARGS__)

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
#define KVIKIO_NVTX_MARKER(message, payload) KVIKIO_NVTX_MARKER_IMPL(message, payload)

}  // namespace kvikio
