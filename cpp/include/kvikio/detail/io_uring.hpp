/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <liburing.h>
#include <linux/io_uring.h>

namespace kvikio::detail {

class IoUringManager {
 private:
  IoUringManager();
  io_uring _ring{};
  unsigned int _queue_depth{};
  std::size_t _chunk_size{};

 public:
  static IoUringManager& get();
  ~IoUringManager() noexcept;

  IoUringManager(IoUringManager const&)            = delete;
  IoUringManager& operator=(IoUringManager const&) = delete;
  IoUringManager(IoUringManager&&)                 = delete;
  IoUringManager& operator=(IoUringManager&&)      = delete;

  io_uring* ring() noexcept;
  unsigned int queue_depth() noexcept;
  std::size_t chunk_size() noexcept;
};

bool is_io_uring_supported() noexcept;

std::size_t io_uring_read_host(int fd, void* buf, std::size_t size, std::size_t file_offset);

std::size_t io_uring_read_device(int fd, void* buf, std::size_t size, std::size_t file_offset);

}  // namespace kvikio::detail