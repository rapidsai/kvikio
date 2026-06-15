/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstring>
#include <string>

#include <curl/curl.h>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

void reset_callback_context(CallbackContext& ctx) noexcept
{
  ctx.offset         = 0;
  ctx.overflow_error = false;
}

bool callback_context_complete(CallbackContext const& ctx) noexcept
{
  return !ctx.overflow_error && ctx.offset >= 0 && static_cast<std::size_t>(ctx.offset) == ctx.size;
}

std::size_t callback_context_received_bytes(CallbackContext const& ctx) noexcept
{
  return ctx.offset < 0 ? 0 : static_cast<std::size_t>(ctx.offset);
}

void expect_callback_context_complete(CallbackContext const& ctx)
{
  KVIKIO_EXPECT(callback_context_complete(ctx),
                "Remote read returned fewer bytes than requested",
                std::runtime_error);
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
  std::memcpy(ctx->buf + ctx->offset, data, nbytes);
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
