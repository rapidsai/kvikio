/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

/**
 * @brief Manages a rotating set of bounce buffers for overlapping network I/O with H2D transfers.
 *
 * This class implements k-way buffering, rotating through buffers circularly: while one buffer
 * receives data from the network, previously filled buffers can be asynchronously copied to device
 * memory. When all buffers have been used, the class synchronizes the CUDA stream before reusing
 * buffers.
 */
class BounceBufferManager {
 public:
  /**
   * @brief Construct a BounceBufferManager with the specified number of bounce buffers.
   *
   * @param num_bounce_buffers Number of bounce buffers to allocate from the pool.
   */
  BounceBufferManager(std::size_t num_bounce_buffers);

  /**
   * @brief Get a pointer to the current bounce buffer's data.
   *
   * @return Pointer to the current buffer's memory.
   */
  void* data() const noexcept;

  /**
   * @brief Copy data from the current bounce buffer to device memory and rotate to the next buffer.
   *
   * Issues an asynchronous H2D copy and advances to the next buffer. When wrapping around to buffer
   * 0, synchronizes the stream to ensure all previous copies have completed before reuse.
   *
   * @param dst Device memory destination pointer.
   * @param size Number of bytes to copy.
   * @param stream CUDA stream for the asynchronous copy.
   * @exception kvikio::CUfileException if size exceeds bounce buffer capacity.
   */
  void copy(void* dst, std::size_t size, CUstream stream);

 private:
  std::size_t _bounce_buffer_idx{};
  std::size_t _num_bounce_buffers{};
  std::vector<CudaPinnedBounceBufferPool::Buffer> _bounce_buffers;
};

/**
 * @brief Context for tracking the state of a single chunked transfer.
 *
 * Each concurrent connection has an associated TransferContext that tracks the destination buffer,
 * transfer progress, and manages optional bounce buffers for GPU destinations.
 */
struct TransferContext {
  bool overflow_error{};
  bool is_host_mem{};
  char* buf{};
  CurlHandle* curl_easy_handle{};
  std::size_t chunk_size{};
  std::size_t bytes_transferred{};
  std::optional<BounceBufferManager> _bounce_buffer_manager;
};

/**
 * @brief Poll-based remote file handle using libcurl's multi interface.
 *
 * This class provides an alternative to the thread-pool-based remote I/O by using libcurl's multi
 * interface with curl_multi_poll() for managing concurrent connections. It implements chunked
 * parallel downloads with k-way buffering to overlap network transfers with host-to-device memory
 * copies.
 *
 * @note Thread safety: The pread() method is protected by a mutex, making it safe to call from
 * multiple threads, though calls will be serialized.
 */
class RemoteHandlePollBased {
 private:
  CURLM* _multi;
  std::size_t _max_connections;
  std::vector<std::unique_ptr<CurlHandle>> _curl_easy_handles;
  std::vector<TransferContext> _transfer_ctxs;
  RemoteEndpoint* _endpoint;
  mutable std::mutex _mutex;

 public:
  /**
   * @brief Construct a poll-based remote handle.
   *
   * Initializes the libcurl multi handle and creates the specified number of easy handles for
   * concurrent transfers.
   *
   * @param endpoint Non-owning pointer to the remote endpoint. Must outlive this object.
   * @param max_connections Maximum number of concurrent connections to use.
   * @exception kvikio::CUfileException if task_size exceeds bounce_buffer_size.
   * @exception kvikio::CUfileException if libcurl multi initialization fails.
   */
  RemoteHandlePollBased(RemoteEndpoint* endpoint, std::size_t max_connections);

  /**
   * @brief Destructor that cleans up libcurl multi resources.
   *
   * Removes all easy handles from the multi handle and performs cleanup. Errors during cleanup are
   * logged but do not throw.
   */
  ~RemoteHandlePollBased();

  /**
   * @brief Read data from the remote file into a buffer.
   *
   * Performs a parallel chunked read using multiple concurrent HTTP range requests. For device
   * memory destinations, uses bounce buffers with k-way buffering to overlap network I/O with H2D
   * transfers.
   *
   * @param buf Destination buffer (host or device memory).
   * @param size Number of bytes to read.
   * @param file_offset Offset in the remote file to start reading from.
   * @return Number of bytes actually read.
   * @exception std::overflow_error if the server returns more data than expected (may indicate the
   * server doesn't support range requests).
   * @exception std::runtime_error on libcurl errors.
   */
  std::size_t pread(void* buf, std::size_t size, std::size_t file_offset = 0);
};
}  // namespace kvikio::detail
