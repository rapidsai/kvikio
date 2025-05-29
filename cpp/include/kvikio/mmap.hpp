/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstddef>
#include <future>

#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>

namespace kvikio {

class MmapHandle {
 private:
  void* _buf{};
  void* _external_buf{};
  std::size_t _initial_size{};
  std::size_t _initial_file_offset{};
  std::size_t _file_size{};

  std::size_t _map_offset{};
  std::size_t _map_size{};
  void* _map_addr{};

  std::ptrdiff_t _offset_delta{};

  bool _initialized{};
  int _map_protection_flag{};
  FileWrapper _file_wrapper{};

  void map();
  void unmap();

  bool has_external_buf() const noexcept;

  std::tuple<void*, void*, std::size_t, std::size_t> prepare_read(std::size_t size,
                                                                  std::size_t file_offset);

 public:
  MmapHandle(std::string const& file_path,
             std::string const& flags        = "r",
             std::size_t initial_size        = 0,
             std::size_t initial_file_offset = 0,
             void* external_buf              = nullptr,
             mode_t mode                     = FileHandle::m644);

  MmapHandle(MmapHandle const&)            = delete;
  MmapHandle& operator=(MmapHandle const&) = delete;
  MmapHandle(MmapHandle&& o) noexcept;
  MmapHandle& operator=(MmapHandle&& o) noexcept;
  ~MmapHandle() noexcept;

  std::size_t requested_size() const noexcept;

  [[nodiscard]] bool closed() const noexcept;

  void close() noexcept;

  std::pair<void*, std::size_t> read(std::size_t size,
                                     std::size_t file_offset = 0,
                                     bool prefault           = false);

  std::pair<void*, std::future<std::size_t>> pread(
    std::size_t size,
    std::size_t file_offset       = 0,
    bool prefault                 = false,
    std::size_t aligned_task_size = defaults::task_size());

  static std::size_t perform_prefault(void* buf, std::size_t size);

  static std::future<std::size_t> perform_prefault_parallel(
    void* buf,
    std::size_t size,
    std::size_t aligned_task_size = defaults::task_size(),
    std::uint64_t call_idx        = 0,
    nvtx_color_type nvtx_color    = NvtxManager::default_color());
};
}  // namespace kvikio
