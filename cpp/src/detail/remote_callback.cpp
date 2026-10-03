/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

#include <curl/curl.h>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>

namespace kvikio::detail {

void CallbackContext::reset_for_retry() noexcept
{
  offset         = 0;
  overflow_error = false;
  segment_index  = 0;
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

  // Easy thread pool backend
  if (ctx->segments.empty()) {
    std::memcpy(ctx->buf + ctx->offset, data, nbytes);
    ctx->offset += nbytes;
    return nbytes;
  }

  // Multi-poll backend
  // `ctx->offset` is the transfer offset for the sub-range updated per write callback invocation.
  // `span_offset` is the transfer offset for the segment determined prior to the transfer.
  // One character is 5 bytes. `#` is a wanted byte and `.` is a gap byte.
  // 1 transfer     ############..........######..........##########
  //                S0          gap       S1    gap       S2
  //                0           60        110   140       190       240
  //
  // Earlier chunks delivered [0, 50), so `ctx->offset` is 50. This chunk delivers [50, 130).
  //
  // this chunk               ##..........####
  //                          50              130
  //
  // - step 1  S0 is current. `filled` = 50 (S0 bytes already written), `copied` = 10 (the rest of
  //           S0). S0 is complete, so `segment_index` moves to S1.
  // - step 2  S1 is current, but `ctx->offset` (60) is before its start (110). `skipped` = 50.
  // - step 3  S1 is current. `filled` = 0, `copied` = 20. S1 is not complete, so `segment_index`
  //           stays.
  // - `remaining` is 0, so the loop ends with `ctx->offset` at 130, where the next chunk starts (at
  //   the next write callback invocation).
  auto remaining = nbytes;
  auto* src      = data;
  while (remaining > 0 && ctx->segment_index < ctx->segments.size()) {
    auto const& segment   = ctx->segments[ctx->segment_index];
    auto const ctx_offset = static_cast<std::size_t>(ctx->offset);

    // In a gap. Advance without copying.
    if (ctx_offset < segment.span_offset) {
      auto const skipped = std::min(remaining, segment.span_offset - ctx_offset);
      src += skipped;
      ctx->offset += static_cast<std::ptrdiff_t>(skipped);
      remaining -= skipped;
      continue;
    }

    auto const filled = ctx_offset - segment.span_offset;
    auto const copied = std::min(remaining, segment.length - filled);
    std::memcpy(static_cast<std::byte*>(segment.buf) + filled, src, copied);
    src += copied;
    ctx->offset += static_cast<std::ptrdiff_t>(copied);
    remaining -= copied;
    if (filled + copied == segment.length) { ++ctx->segment_index; }
  }

  // The tail is unwanted bytes.
  ctx->offset += static_cast<std::ptrdiff_t>(remaining);
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
