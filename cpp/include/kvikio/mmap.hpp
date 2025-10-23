/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 *
 * This utility class facilitates the use of file-backed memory by providing a performant method
 * `pread()` to read a range of data into user-provided memory residing on the host or device.
 *
 * File-backed memory can be considered when a large number of nonadjacent file ranges (specified by
 * the `offset` and `size` pair) are to be frequently accessed. It can potentially reduce memory
 * usage due to demand paging (compared to reading the entire file with `read(2)`), and may improve
 * I/O performance compared to frequent calls to `read(2)`.
 */
class MmapHandle {
 private:
  void* _buf{};
  std::size_t _initial_map_size{};
  std::size_t _initial_map_offset{};
  std::size_t _file_size{};
  std::size_t _map_offset{};
  std::size_t _map_size{};
  void* _map_addr{};
  bool _initialized{};
  int _map_protection{};
  int _map_flags{};
  FileWrapper _file_wrapper{};

  /**
   * @brief Validate and adjust the read arguments.
   *
   * @param size Size in bytes to read. If not specified, set it to the bytes from `offset` to
   * the end of file
   * @param offset File offset
   * @return Adjusted size in bytes to read
   *
   * @exception std::out_of_range if the read region specified by `offset` and `size` is
   * outside the initial region specified when the mapping handle was constructed
   * @exception std::runtime_error if the mapping handle is closed
   */
  std::size_t validate_and_adjust_read_args(std::optional<std::size_t> const& size,
                                            std::size_t offset);

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
   * @param initial_map_size Size in bytes of the mapped region. Must be greater than 0. If not
   * specified, map the region starting from `initial_map_offset` to the end of file
   * @param initial_map_offset File offset of the mapped region
   * @param mode Access mode
   * @param map_flags Flags to be passed to the system call `mmap`. See `mmap(2)` for details
   * @exception std::out_of_range if `initial_map_offset` (left bound of the mapped region) is equal
   * to or greater than the file size
   * @exception std::out_of_range if the sum of `initial_map_offset` and `initial_map_size` (right
   * bound of the mapped region) is greater than the file size
   * @exception std::invalid_argument if `initial_map_size` is given but is 0
   */
  MmapHandle(std::string const& file_path,
             std::string const& flags                    = "r",
             std::optional<std::size_t> initial_map_size = std::nullopt,
             std::size_t initial_map_offset              = 0,
             mode_t mode                                 = FileHandle::m644,
             std::optional<int> map_flags                = std::nullopt);

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
  [[nodiscard]] std::size_t initial_map_size() const noexcept;

  /**
   * @brief File offset of the mapped region when the mapping handle was constructed
   *
   * @return Initial file offset of the mapped region
   */
  [[nodiscard]] std::size_t initial_map_offset() const noexcept;

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
   * @brief Sequential read `size` bytes from the file (with the offset `offset`) to the
   * destination buffer `buf`
   *
   * @param buf Address of the host or device memory (destination buffer)
   * @param size Size in bytes to read. Can be 0 in which case nothing will be read. If not
   * specified, read starts from `offset` to the end of file
   * @param offset File offset
   * @return Number of bytes that have been read
   *
   * @exception std::out_of_range if the read region specified by `offset` and `size` is
   * outside the initial region specified when the mapping handle was constructed
   * @exception std::runtime_error if the mapping handle is closed
   */
  std::size_t read(void* buf,
                   std::optional<std::size_t> size = std::nullopt,
                   std::size_t offset              = 0);

  /**
   * @brief Parallel read `size` bytes from the file (with the offset `offset`) to the
   * destination buffer `buf`
   *
   * @param buf Address of the host or device memory (destination buffer)
   * @param size Size in bytes to read. Can be 0 in which case nothing will be read. If not
   * specified, read starts from `offset` to the end of file
   * @param offset File offset
   * @param task_size Size of each task in bytes
   * @return Future that on completion returns the size of bytes that were successfully read.
   *
   * @exception std::out_of_range if the read region specified by `offset` and `size` is
   * outside the initial region specified when the mapping handle was constructed
   * @exception std::runtime_error if the mapping handle is closed
   *
   * @note The `std::future` object's `wait()` or `get()` should not be called after the lifetime of
   * the MmapHandle object ends. Otherwise, the behavior is undefined.
   */
  std::future<std::size_t> pread(void* buf,
                                 std::optional<std::size_t> size = std::nullopt,
                                 std::size_t offset              = 0,
                                 std::size_t task_size           = defaults::task_size());
};

}  // namespace kvikio
