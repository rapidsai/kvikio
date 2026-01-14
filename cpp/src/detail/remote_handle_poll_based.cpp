/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/detail/remote_handle_poll_based.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {
namespace {
std::size_t callback_memory(char* buffer, std::size_t size, std::size_t nmemb, void* userdata)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto* ctx                = reinterpret_cast<TransferContext*>(userdata);
  std::size_t const nbytes = size * nmemb;

  if (ctx->chunk_size < ctx->bytes_transferred + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);
  void* dst = ctx->is_host_mem ? ctx->buf : ctx->bounce_buffer.get();
  std::memcpy(static_cast<char*>(dst) + ctx->bytes_transferred, buffer, nbytes);
  ctx->bytes_transferred += nbytes;
  return nbytes;
}

void reconfig_easy_handle(CurlHandle& curl_easy_handle,
                          TransferContext* ctx,
                          void* buf,
                          bool is_host_mem,
                          std::size_t chunk_idx,
                          std::size_t chunk_size,
                          std::size_t size,
                          std::size_t offset)
{
  auto const local_offset      = chunk_idx * chunk_size;
  auto const actual_chunk_size = std::min(chunk_size, size - local_offset);

  ctx->overflow_error    = false;
  ctx->is_host_mem       = is_host_mem;
  ctx->buf               = static_cast<char*>(buf) + local_offset;
  ctx->chunk_size        = actual_chunk_size;
  ctx->bytes_transferred = 0;

  detail::setup_range_request_impl(curl_easy_handle, offset + local_offset, actual_chunk_size);
};
}  // namespace

TransferContext::TransferContext() : bounce_buffer{CudaPinnedBounceBufferPool::instance().get()} {}

RemoteHandlePollBased::RemoteHandlePollBased(std::string const& url,
                                             RemoteEndpoint* endpoint,
                                             std::size_t num_conns)
  : _url{url}, _endpoint{endpoint}, _num_conns{num_conns}, _transfer_ctxs(_num_conns)
{
  _multi = curl_multi_init();
  KVIKIO_EXPECT(_multi != nullptr, "Failed to initialize libcurl multi API");

  for (std::size_t i = 0; i < _num_conns; ++i) {
    _curl_easy_handles.emplace_back(
      std::make_unique<CurlHandle>(kvikio::LibCurl::instance().get_handle(),
                                   kvikio::detail::fix_conda_file_path_hack(__FILE__),
                                   KVIKIO_STRINGIFY(__LINE__)));
    _endpoint->setopt(*_curl_easy_handles.back());
    _transfer_ctxs.emplace_back();
  }
}

RemoteHandlePollBased::~RemoteHandlePollBased()
{
  try {
    KVIKIO_CHECK_CURL_MULTI(curl_multi_cleanup(_multi));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(e.what());
  }
}

std::size_t RemoteHandlePollBased::pread(void* buf, std::size_t size, std::size_t file_offset)
{
  if (size == 0) return 0;

  std::size_t const chunk_size = defaults::task_size();
  std::size_t num_chunks       = (size + chunk_size - 1) / chunk_size;
  std::size_t actual_num_conns = std::min(_num_conns, num_chunks);

  return 123;
}
}  // namespace kvikio::detail
