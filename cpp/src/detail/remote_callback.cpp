/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include <curl/curl.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>

namespace kvikio::detail {

namespace {
/**
 * @brief Whether `KVIKIO_REMOTE_IO_DISCARD_DATA` is enabled.
 *
 * Drop received data instead of copying it into host memory, to benchmark the network path alone.
 * Reads into device memory are not affected. The destination buffer is left untouched, with no
 * error raised. Do not enable outside a benchmark.
 */
bool discard_data_enabled()
{
  static bool const value = getenv_or("KVIKIO_REMOTE_IO_DISCARD_DATA", false);
  return value;
}

/**
 * @brief Whether `KVIKIO_REMOTE_IO_NONTEMPORAL_COPY` is enabled.
 *
 * Copy received data into host memory with non-temporal stores, which skip fetching the
 * destination cache lines. This includes the pinned bounce buffers of device reads.
 *
 * It helps only when the destination is much larger than the last-level cache and is not read
 * again soon. It requires x86-64 with AVX2, and falls back to `memcpy` elsewhere.
 */
bool nontemporal_copy_enabled()
{
  static bool const value = getenv_or("KVIKIO_REMOTE_IO_NONTEMPORAL_COPY", false);
  return value;
}

/**
 * @brief Whether the CPU supports AVX2. Always false on non-x86-64 targets.
 */
bool cpu_supports_avx2()
{
#if defined(__x86_64__)
  // `__builtin_cpu_supports` is an x86 built-in function, used here to check at **runtime** if the
  // machine supports AVX2
  static bool const value = __builtin_cpu_supports("avx2");
  return value;
#else
  return false;
#endif
}

/**
 * @brief Copy with AVX2 non-temporal stores on x86-64, and `memcpy` elsewhere. On x86-64, call only
 * when `cpu_supports_avx2()` is true.
 */
#if defined(__x86_64__)
// Compile with AVX2 for this function alone (regardless of whether the x86-64 machine at
// **compile-time** supports AVX2 or not), as if by -mavx2. The rest of the library is compiled with
// baseline x86-64 options.
[[gnu::target("avx2")]] void copy_nontemporal_impl(std::byte* dst,
                                                   std::byte const* src,
                                                   std::size_t nbytes)
{
  // Non-temporal stores need a 32-byte aligned destination.
  constexpr std::size_t alignment = 32;
  auto const misaligned           = reinterpret_cast<std::uintptr_t>(dst) % alignment;
  if (misaligned != 0) {
    auto const head = std::min(nbytes, alignment - misaligned);
    std::memcpy(dst, src, head);
    dst += head;
    src += head;
    nbytes -= head;
  }
  while (nbytes >= alignment) {
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),
                        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(src)));
    dst += alignment;
    src += alignment;
    nbytes -= alignment;
  }
  if (nbytes != 0) { std::memcpy(dst, src, nbytes); }
  // Non-temporal stores are weakly ordered.
  _mm_sfence();
}
#else
void copy_nontemporal_impl(std::byte* dst, std::byte const* src, std::size_t nbytes)
{
  std::memcpy(dst, src, nbytes);
}
#endif

}  // namespace

void copy_nontemporal(std::byte* dst, std::byte const* src, std::size_t nbytes)
{
  if (cpu_supports_avx2()) {
    copy_nontemporal_impl(dst, src, nbytes);
  } else {
    std::memcpy(dst, src, nbytes);
  }
}

void copy_received_data(std::byte* dst, std::byte const* src, std::size_t nbytes)
{
  if (nontemporal_copy_enabled()) {
    copy_nontemporal(dst, src, nbytes);
  } else {
    std::memcpy(dst, src, nbytes);
  }
}

void CallbackContext::reset_for_retry() noexcept
{
  offset         = 0;
  overflow_error = false;
}

std::size_t callback_host_memory(char* data, std::size_t size, std::size_t nmemb, void* context)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);
  if (!discard_data_enabled()) {
    copy_received_data(reinterpret_cast<std::byte*>(ctx->buf + ctx->offset),
                       reinterpret_cast<std::byte const*>(data),
                       nbytes);
  }
  ctx->offset += nbytes;
  return nbytes;
}

std::size_t callback_pinned_buffer(char* data, std::size_t size, std::size_t nmemb, void* context)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);
  copy_received_data(static_cast<std::byte*>(ctx->pinned_buffer) + ctx->offset,
                     reinterpret_cast<std::byte const*>(data),
                     nbytes);
  ctx->offset += nbytes;
  return nbytes;
}

std::size_t callback_get_string_response(char* data,
                                         std::size_t size,
                                         std::size_t num_bytes,
                                         void* userdata)
{
  auto new_data_size = size * num_bytes;
  auto* response     = reinterpret_cast<std::string*>(userdata);
  response->append(data, new_data_size);
  return new_data_size;
}
}  // namespace kvikio::detail
