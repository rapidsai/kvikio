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
#include <immintrin.h>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>

namespace kvikio::detail {

void CallbackContext::reset_for_retry() noexcept
{
  offset         = 0;
  overflow_error = false;
}

// Benchmarking escape hatch: accept and account for every byte without
// delivering it. The point is to separate the cost of moving bytes off the
// network from the cost of landing them in a caller buffer, which at high
// concurrency is a stream of writes to memory far larger than L3 and so is not
// free. Byte accounting is unchanged, so a throughput figure measured with this
// on is still an honest count of bytes received -- but the destination holds
// garbage, which is why it is off unless explicitly asked for.
//
// Read once: this callback runs per libcurl buffer, millions of times a second.
static bool const discard_payload = [] {
  auto const* env = std::getenv("KVIKIO_REMOTE_IO_DISCARD");
  return env != nullptr && env[0] == '1';
}();

// Non-temporal copy for the receive path.
//
// libcurl delivers a buffer at a time, 16 KiB by default, so every memcpy here
// is far below glibc's non-temporal threshold and uses ordinary stores. Ordinary
// stores to a destination much larger than last-level cache fetch each line
// first (read-for-ownership), so a destination byte costs two DRAM accesses
// rather than one, and at hundreds of Gbps the receive path becomes
// memory-bandwidth-bound rather than CPU-bound. Streaming stores skip the fetch.
//
// Only worthwhile when the destination is not read again soon, which is exactly
// the case for a buffer that is about to be handed to a GPU or a consumer
// thread. Off by default because it is the wrong choice for a small destination
// that stays in cache.
static bool const nt_copy = [] {
  auto const* env = std::getenv("KVIKIO_REMOTE_IO_NT_COPY");
  return env != nullptr && env[0] == '1';
}();

namespace {

#if defined(__AVX512F__) || defined(__AVX2__)
void copy_nontemporal(char* dst, char const* src, std::size_t nbytes)
{
  // Streaming stores need a 32-byte aligned destination, so copy the leading
  // partial line normally and resume streaming once aligned.
  constexpr std::size_t kVec = 32;
  auto const misaligned      = reinterpret_cast<std::uintptr_t>(dst) % kVec;
  if (misaligned != 0) {
    auto const head = std::min(nbytes, kVec - misaligned);
    std::memcpy(dst, src, head);
    dst += head;
    src += head;
    nbytes -= head;
  }
  while (nbytes >= kVec) {
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),
                        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(src)));
    dst += kVec;
    src += kVec;
    nbytes -= kVec;
  }
  if (nbytes != 0) { std::memcpy(dst, src, nbytes); }
  // Streaming stores are weakly ordered with respect to everything else, so the
  // writes must be fenced before the buffer is handed on.
  _mm_sfence();
}
#else
void copy_nontemporal(char* dst, char const* src, std::size_t nbytes)
{
  std::memcpy(dst, src, nbytes);
}
#endif

}  // namespace

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
  if (!discard_payload) {
    if (nt_copy) {
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
