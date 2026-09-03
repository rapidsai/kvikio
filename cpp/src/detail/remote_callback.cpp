/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include <curl/curl.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define KVIKIO_X86_INTRINSICS
#endif

#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>

namespace kvikio::detail {

namespace {
/**
 * @brief Whether the remote-to-host copy is skipped, controlled by the environment variable
 * `KVIKIO_REMOTE_NO_HOST_COPY`. Benchmark only. The destination buffer is left untouched. Applies
 * to both backends, and host destinations only.
 *
 * @return True when the copy is skipped.
 */
bool no_host_copy_enabled()
{
  static bool const value = getenv_or("KVIKIO_REMOTE_NO_HOST_COPY", false);
  return value;
}

/**
 * @brief Whether the remote-to-host copy uses non-temporal (streaming) stores, controlled by the
 * environment variable `KVIKIO_REMOTE_NONTEMPORAL_COPY`.
 *
 * Each libcurl write callback delivers at most `CURL_MAX_WRITE_SIZE` (16 KiB), below glibc's
 * memcpy threshold for switching to non-temporal stores on its own. As a result by default every
 * callback copy uses ordinary stores, which first fetch the destination cache line before writing
 * it, costing two DRAM accesses per byte instead of one. Non-temporal stores skip that fetch, but
 * only pay off when the destination is much larger than last-level cache and is not read again
 * soon, since the write bypasses the cache entirely.
 *
 * @return True when non-temporal stores are used.
 */
bool nontemporal_copy_enabled()
{
  static bool const value = getenv_or("KVIKIO_REMOTE_NONTEMPORAL_COPY", false);
  return value;
}

#if defined(KVIKIO_X86_INTRINSICS) && (defined(__GNUC__) || defined(__clang__))
[[gnu::target("avx2")]] void copy_nontemporal_avx2(char* dst, char const* src, std::size_t nbytes)
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

void copy_nontemporal(char* dst, char const* src, std::size_t nbytes)
{
  // Runtime CPU feature check
  static bool const has_avx2 = __builtin_cpu_supports("avx2");
  if (has_avx2) {
    copy_nontemporal_avx2(dst, src, nbytes);
  } else {
    std::memcpy(dst, src, nbytes);
  }
}
#else
void copy_nontemporal(char* dst, char const* src, std::size_t nbytes)
{
  std::memcpy(dst, src, nbytes);
}
#endif

}  // namespace

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
  if (!no_host_copy_enabled()) {
    if (nontemporal_copy_enabled()) {
      copy_nontemporal(ctx->buf + ctx->offset, data, nbytes);
    } else {
      std::memcpy(ctx->buf + ctx->offset, data, nbytes);
    }
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
  std::memcpy(static_cast<char*>(ctx->pinned_buffer) + ctx->offset, data, nbytes);
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
