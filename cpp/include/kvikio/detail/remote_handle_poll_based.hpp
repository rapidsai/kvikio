/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

#define KVIKIO_CHECK_CURL_EASY(err_code) \
  kvikio::detail::check_curl_easy(err_code, __FILE__, __LINE__)

#define KVIKIO_CHECK_CURL_MULTI(err_code) \
  kvikio::detail::check_curl_multi(err_code, __FILE__, __LINE__)

namespace kvikio::detail {

inline void check_curl_easy(CURLcode err_code, char const* filename, int line_number)
{
  if (err_code == CURLcode::CURLE_OK) { return; }
  std::stringstream ss;
  ss << "libcurl error: " << curl_easy_strerror(err_code) << " at: " << filename << ":"
     << line_number << "\n";
  throw std::runtime_error(ss.str());
}

inline void check_curl_multi(CURLMcode err_code, char const* filename, int line_number)
{
  if (err_code == CURLMcode::CURLM_OK) { return; }
  std::stringstream ss;
  ss << "libcurl error: " << curl_multi_strerror(err_code) << " at: " << filename << ":"
     << line_number << "\n";
  throw std::runtime_error(ss.str());
}

class BounceBufferManager {
 public:
  BounceBufferManager(std::size_t num_bounce_buffers);

  void* data() const noexcept;

  void copy(void* dst, std::size_t size, CUstream stream);

 private:
  std::size_t _bounce_buffer_idx{};
  std::size_t _num_bounce_buffers{};
  std::vector<CudaPinnedBounceBufferPool::Buffer> _bounce_buffers;
};

struct TransferContext {
  bool overflow_error{};
  bool is_host_mem{};
  char* buf{};
  std::size_t chunk_size{};
  std::size_t bytes_transferred{};
  std::optional<BounceBufferManager> _bounce_buffer_manager;
};

class RemoteHandlePollBased {
 private:
  CURLM* _multi;
  std::size_t _max_connections;
  std::vector<std::unique_ptr<CurlHandle>> _curl_easy_handles;
  std::vector<TransferContext> _transfer_ctxs;
  RemoteEndpoint* _endpoint;
  mutable std::mutex _mutex;

 public:
  RemoteHandlePollBased(RemoteEndpoint* endpoint, std::size_t max_connections);

  ~RemoteHandlePollBased();

  std::size_t pread(void* buf, std::size_t size, std::size_t file_offset = 0);
};
}  // namespace kvikio::detail
