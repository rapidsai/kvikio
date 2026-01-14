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
#include <kvikio/utils.hpp>

namespace kvikio::detail {
namespace {
std::size_t write_callback(char* buffer, std::size_t size, std::size_t nmemb, void* userdata)
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

void reconfig_easy_handle(CURL* curl_easy_handle,
                          TransferContext* ctx,
                          void* buf,
                          bool is_host_mem,
                          std::size_t current_chunk_idx,
                          std::size_t chunk_size,
                          std::size_t size,
                          std::size_t file_offset)
{
  auto const local_offset      = current_chunk_idx * chunk_size;
  auto const actual_chunk_size = std::min(chunk_size, size - local_offset);

  ctx->overflow_error    = false;
  ctx->is_host_mem       = is_host_mem;
  ctx->buf               = static_cast<char*>(buf) + local_offset;
  ctx->chunk_size        = actual_chunk_size;
  ctx->bytes_transferred = 0;

  std::size_t const remote_start = file_offset + local_offset;
  std::size_t const remote_end   = remote_start + actual_chunk_size - 1;
  std::string const byte_range   = std::to_string(remote_start) + "-" + std::to_string(remote_end);
  KVIKIO_CHECK_CURL_EASY(curl_easy_setopt(curl_easy_handle, CURLOPT_RANGE, byte_range.c_str()));
};
}  // namespace

TransferContext::TransferContext() : bounce_buffer{CudaPinnedBounceBufferPool::instance().get()} {}

RemoteHandlePollBased::RemoteHandlePollBased(RemoteEndpoint* endpoint, std::size_t max_connections)
  : _endpoint{endpoint}, _max_connections{max_connections}, _transfer_ctxs(_max_connections)
{
  _multi = curl_multi_init();
  KVIKIO_EXPECT(_multi != nullptr, "Failed to initialize libcurl multi API");

  _curl_easy_handles.reserve(_max_connections);
  for (std::size_t i = 0; i < _max_connections; ++i) {
    _curl_easy_handles.emplace_back(
      std::make_unique<CurlHandle>(kvikio::LibCurl::instance().get_handle(),
                                   kvikio::detail::fix_conda_file_path_hack(__FILE__),
                                   KVIKIO_STRINGIFY(__LINE__)));

    // Initialize easy handle, associate it with transfer context
    _endpoint->setopt(*_curl_easy_handles.back());
    _curl_easy_handles.back()->setopt(CURLOPT_WRITEFUNCTION, write_callback);
    _curl_easy_handles.back()->setopt(CURLOPT_WRITEDATA, &_transfer_ctxs[i]);
    _curl_easy_handles.back()->setopt(CURLOPT_PRIVATE, &_transfer_ctxs[i]);
  }
}

RemoteHandlePollBased::~RemoteHandlePollBased()
{
  try {
    // Remove any lingering handles before cleanup
    for (auto& handle : _curl_easy_handles) {
      // Ignore errors
      KVIKIO_CHECK_CURL_MULTI(curl_multi_remove_handle(_multi, handle->handle()));
    }
    KVIKIO_CHECK_CURL_MULTI(curl_multi_cleanup(_multi));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(e.what());
  }
}

std::size_t RemoteHandlePollBased::pread(void* buf, std::size_t size, std::size_t file_offset)
{
  if (size == 0) return 0;

  bool const is_host_mem = is_host_memory(buf);

  auto const chunk_size             = defaults::task_size();
  auto const num_chunks             = (size + chunk_size - 1) / chunk_size;
  auto const actual_max_connections = std::min(_max_connections, num_chunks);

  // Prepare for the run
  std::size_t num_byte_transferred{0};
  std::size_t current_chunk_idx{0};
  for (std::size_t i = 0; i < actual_max_connections; ++i) {
    reconfig_easy_handle(_curl_easy_handles[i]->handle(),
                         &_transfer_ctxs[i],
                         buf,
                         is_host_mem,
                         current_chunk_idx++,
                         chunk_size,
                         size,
                         file_offset);
    KVIKIO_CHECK_CURL_MULTI(curl_multi_add_handle(_multi, _curl_easy_handles[i]->handle()));
  }

  // Start the run
  int still_running{0};
  do {
    KVIKIO_CHECK_CURL_MULTI(curl_multi_perform(_multi, &still_running));

    CURLMsg* msg;
    int msgs_left;

    while ((msg = curl_multi_info_read(_multi, &msgs_left))) {
      if (msg->msg != CURLMSG_DONE) continue;

      TransferContext* ctx{nullptr};
      KVIKIO_CHECK_CURL_EASY(curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &ctx));

      KVIKIO_EXPECT(msg->data.result == CURLE_OK,
                    "Chunked transfer failed in poll-based multi API");
      num_byte_transferred += ctx->bytes_transferred;
      KVIKIO_CHECK_CURL_MULTI(curl_multi_remove_handle(_multi, msg->easy_handle));

      if (current_chunk_idx < num_chunks) {
        reconfig_easy_handle(msg->easy_handle,
                             ctx,
                             buf,
                             is_host_mem,
                             current_chunk_idx++,
                             chunk_size,
                             size,
                             file_offset);
        KVIKIO_CHECK_CURL_MULTI(curl_multi_add_handle(_multi, msg->easy_handle));
      }
    }

    if (still_running > 0) {
      KVIKIO_CHECK_CURL_MULTI(curl_multi_poll(_multi, nullptr, 0, 1000, nullptr));
    }

  } while (still_running > 0);

  return num_byte_transferred;
}
}  // namespace kvikio::detail
