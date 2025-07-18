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
#include <optional>

namespace kvikio {

/**
 * @brief Handle of a memory-mapped file
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
  int _map_protection{};
  int _map_flags{};
  FileWrapper _file_wrapper{};

  /**
   * @brief For the specified memory range, touch the first byte of each page to cause page fault.
   *
   * For the first page, if the starting address is not aligned to the page boundary, the byte at
   * that address is touched.
   *
   * @param buf The starting memory address
   * @param size The size in bytes of the memory range
   * @return The number of bytes touched
   */
  std::size_t perform_prefault(void* buf, std::size_t size);

  /**
   * @brief Validate and adjust the read arguments.
   *
   * @param size Size in bytes to read. If not specified, set it to the bytes from `file_offset` to
   * the end of file
   * @param file_offset File offset
   * @return Adjusted size in bytes to read
   *
   * @exception std::out_of_range if the read region specified by `file_offset` and `size` is
   * outside the initial region specified when the mapping handle was constructed
   * @exception std::invalid_argument if the size is given but is 0
   * @exception std::runtime_error if the mapping handle is closed
   */
  std::size_t validate_and_adjust_read_args(std::optional<std::size_t> const& size,
                                            std::size_t& file_offset);

  /**
   * @brief Implementation of read
   *
   * Copy data from the source buffer `global_src_buf + buf_offset` to the destination buffer
   * `global_dst_buf + buf_offset`.
   *
   * @param global_dst_buf Address of the host or device memory (destination buffer)
   * @param global_src_buf Address of the host memory (source buffer)
   * @param size Size in bytes to read
   * @param buf_offset Offset for both `global_dst_buf` and `global_src_buf`
   * @param is_dst_buf_host_mem Whether the destination buffer is host memory or not
   * @param ctx CUDA context when the destination buffer is not host memory
   */
  void read_impl(void* global_dst_buf,
                 void* global_src_buf,
                 std::size_t size,
                 std::size_t buf_offset,
                 bool is_dst_buf_host_mem,
                 CUcontext ctx);

 public:
  /**
   * @brief Construct an empty memory-mapped file
   *
   */
  MmapHandle() noexcept = default;

  /**
   * @brief Construct a new memory-mapped file
   *
   * @param file_path File path
   * @param flags Open flags (see also `fopen(3)`):
   *   - "r": "open for reading (default)"
   *   - "w": "open for writing, truncating the file first"
   *   - "a": "open for writing, appending to the end of file if it exists"
   *   - "+": "open for updating (reading and writing)"
   * @param initial_size Size in bytes of the mapped region. If not specified, map the region
   * starting from `initial_file_offset` to the end of file
   * @param initial_file_offset File offset of the mapped region
   * @param mode Access mode
   * @param map_flags Flags to be passed to the system call `mmap`. See `mmap(2)` for details
   */
  MmapHandle(std::string const& file_path,
             std::string const& flags                = "r",
             std::optional<std::size_t> initial_size = std::nullopt,
             std::size_t initial_file_offset         = 0,
             mode_t mode                             = FileHandle::m644,
             std::optional<int> map_flags            = std::nullopt);

  MmapHandle(MmapHandle const&)            = delete;
  MmapHandle& operator=(MmapHandle const&) = delete;
  MmapHandle(MmapHandle&& o) noexcept;
  MmapHandle& operator=(MmapHandle&& o) noexcept;
  ~MmapHandle() noexcept;

  /**
   * @brief Size in bytes of the mapped region when the mapping handle was constructed
   *
   * @return Initial size of the mapped region
   */
  [[nodiscard]] std::size_t initial_size() const noexcept;

  /**
   * @brief File offset of the mapped region when the mapping handle was constructed
   *
   * @return Initial file offset of the mapped region
   */
  [[nodiscard]] std::size_t initial_file_offset() const noexcept;

  /**
   * @brief Get the file size if the file is open. Returns 0 if the file is closed.
   *
   * The behavior of this method is consistent with `FileHandle::nbytes`.
   *
   * @return The file size in bytes
   */
  [[nodiscard]] std::size_t file_size() const;

  /**
   * @brief Alias of `file_size`
   *
   * @return The file size in bytes
   */
  [[nodiscard]] std::size_t nbytes() const;

  /**
   * @brief Whether the mapping handle is closed
   *
   * @return Boolean answer
   */
  [[nodiscard]] bool closed() const noexcept;

  /**
   * @brief Close the mapping handle if it is open; do nothing otherwise
   */
  void close() noexcept;

  /**
   * @brief Sequential read `size` bytes from the file (with the offset `file_offset`) to the
   * destination buffer `buf`
   *
   * @param buf Address of the host or device memory (destination buffer)
   * @param size Size in bytes to read. If not specified, read starts from `file_offset` to the end
   * of file
   * @param file_offset File offset
   * @return Number of bytes that have been read
   *
   * @exception std::out_of_range if the read region specified by `file_offset` and `size` is
   * outside the initial region specified when the mapping handle was constructed
   * @exception std::invalid_argument if the size is given but is 0
   * @exception std::runtime_error if the mapping handle is closed
   */
  std::size_t read(void* buf,
                   std::optional<std::size_t> size = std::nullopt,
                   std::size_t file_offset         = 0);

  /**
   * @brief Parallel read `size` bytes from the file (with the offset `file_offset`) to the
   * destination buffer `buf`
   *
   * @param buf Address of the host or device memory (destination buffer)
   * @param size Size in bytes to read. If not specified, read starts from `file_offset` to the end
   * of file
   * @param file_offset File offset
   * @param task_size Size of each task in bytes
   * @return Future that on completion returns the size of bytes that were successfully read.
   *
   * @exception std::out_of_range if the read region specified by `file_offset` and `size` is
   * outside the initial region specified when the mapping handle was constructed
   * @exception std::invalid_argument if the size is given but is 0
   * @exception std::runtime_error if the mapping handle is closed
   *
   * @note The `std::future` object's `wait()` or `get()` should not be called after the lifetime of
   * the MmapHandle object ends. Otherwise, the behavior is undefined.
   */
  std::future<std::size_t> pread(void* buf,
                                 std::optional<std::size_t> size = std::nullopt,
                                 std::size_t file_offset         = 0,
                                 std::size_t task_size           = defaults::task_size());
};
}  // namespace kvikio
