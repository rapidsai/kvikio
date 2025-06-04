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

/**
 * @brief
 *
 */
class MmapHandle {
 private:
  void* _buf{};
  std::size_t _initial_size{};
  std::size_t _initial_file_offset{};
  std::size_t _file_size{};
  std::size_t _map_offset{};
  std::size_t _map_size{};
  void* _map_addr{};
  bool _initialized{};
  int _map_protection_flags{};
  int _map_core_flags{};
  FileWrapper _file_wrapper{};

  /**
   * @brief
   *
   */
  void map();

  /**
   * @brief
   *
   */
  void unmap();

  /**
   * @brief
   *
   * @param size
   * @param file_offset
   * @return
   */
  std::tuple<void*, void*, void*, std::size_t> prepare_read(std::size_t size,
                                                            std::size_t file_offset);

 public:
  /**
   * @brief Construct a new Mmap Handle object
   *
   */
  MmapHandle() noexcept = default;

  /**
   * @brief Construct a new Mmap Handle object
   *
   * @param file_path
   * @param flags
   * @param initial_size
   * @param initial_file_offset
   * @param mode
   */
  MmapHandle(std::string const& file_path,
             std::string const& flags        = "r",
             std::size_t initial_size        = 0,
             std::size_t initial_file_offset = 0,
             mode_t mode                     = FileHandle::m644);

  MmapHandle(MmapHandle const&)            = delete;
  MmapHandle& operator=(MmapHandle const&) = delete;
  MmapHandle(MmapHandle&& o) noexcept;
  MmapHandle& operator=(MmapHandle&& o) noexcept;
  ~MmapHandle() noexcept;

  /**
   * @brief
   *
   * @return std::size_t
   */
  std::size_t initial_size() const noexcept;

  /**
   * @brief
   *
   * @return Boolean answer
   */
  [[nodiscard]] bool closed() const noexcept;

  /**
   * @brief
   *
   */
  void close() noexcept;

  /**
   * @brief
   *
   * @param size
   * @param file_offset
   * @return
   */
  std::size_t read(void* buf, std::size_t size, std::size_t file_offset = 0);

  /**
   * @brief
   *
   * @param size
   * @param file_offset
   * @param aligned_task_size
   * @return
   */
  std::future<std::size_t> pread(void* buf,
                                 std::size_t size,
                                 std::size_t file_offset = 0,
                                 std::size_t task_size   = defaults::mmap_task_size());
};
}  // namespace kvikio
