/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <optional>
#include <string>

#include <kvikio/shim/cufile_h_wrapper.hpp>

namespace kvikio {
/**
 * @brief Class that provides RAII for file handling.
 */
class FileWrapper {
 private:
  int _fd{-1};

 public:
  /**
   * @brief Open file.
   *
   * @param file_path File path.
   * @param flags Open flags given as a string.
   * @param o_direct Append O_DIRECT to `flags`.
   * @param mode Access modes.
   */
  FileWrapper(std::string const& file_path, std::string const& flags, bool o_direct, mode_t mode);

  /**
   * @brief Construct an empty file wrapper object without opening a file.
   */
  FileWrapper() noexcept = default;

  ~FileWrapper() noexcept;
  FileWrapper(FileWrapper const&)            = delete;
  FileWrapper& operator=(FileWrapper const&) = delete;
  FileWrapper(FileWrapper&& o) noexcept;
  FileWrapper& operator=(FileWrapper&& o) noexcept;

  /**
   * @brief Open file using `open(2)`
   *
   * @param file_path File path.
   * @param flags Open flags given as a string.
   * @param o_direct Append O_DIRECT to `flags`.
   * @param mode Access modes.
   */
  void open(std::string const& file_path, std::string const& flags, bool o_direct, mode_t mode);

  /**
   * @brief Check if the file has been opened.
   *
   * @return A boolean answer indicating if the file has been opened.
   */
  bool opened() const noexcept;

  /**
   * @brief Close the file if it is opened; do nothing otherwise.
   */
  void close() noexcept;

  /**
   * @brief Return the file descriptor.
   *
   * @return File descriptor.
   */
  int fd() const noexcept;
};

/**
 * @brief Class that provides RAII for the cuFile handle.
 */
class CUFileHandleWrapper {
 private:
  CUfileHandle_t _handle{};
  bool _registered{false};

 public:
  CUFileHandleWrapper() noexcept = default;
  ~CUFileHandleWrapper() noexcept;
  CUFileHandleWrapper(CUFileHandleWrapper const&)            = delete;
  CUFileHandleWrapper& operator=(CUFileHandleWrapper const&) = delete;
  CUFileHandleWrapper(CUFileHandleWrapper&& o) noexcept;
  CUFileHandleWrapper& operator=(CUFileHandleWrapper&& o) noexcept;

  /**
   * @brief Register the file handle given the file descriptor.
   *
   * @param fd File descriptor.
   * @return Return the cuFile error code from handle register. If the handle has already been
   * registered by calling `register_handle()`, return `std::nullopt`.
   */
  std::optional<CUfileError_t> register_handle(int fd) noexcept;

  /**
   * @brief Check if the handle has been registered.
   *
   * @return A boolean answer indicating if the handle has been registered.
   */
  bool registered() const noexcept;

  /**
   * @brief Return the cuFile handle.
   *
   * @return The cuFile handle.
   */
  CUfileHandle_t handle() const noexcept;

  /**
   * @brief Unregister the handle if it has been registered; do nothing otherwise.
   */
  void unregister_handle() noexcept;
};

/**
 * @brief Parse open file flags given as a string and return oflags
 *
 * @param flags The flags
 * @param o_direct Append O_DIRECT to the open flags
 * @return oflags
 *
 * @exception std::invalid_argument if the specified flags are not supported.
 * @exception std::invalid_argument if `o_direct` is true, but `O_DIRECT` is not supported.
 */
int open_fd_parse_flags(std::string const& flags, bool o_direct);

/**
 * @brief Open file using `open(2)`
 *
 * @param flags Open flags given as a string
 * @param o_direct Append O_DIRECT to `flags`
 * @param mode Access modes
 * @return File descriptor
 */
int open_fd(std::string const& file_path, std::string const& flags, bool o_direct, mode_t mode);

/**
 * @brief Get the flags of the file descriptor (see `open(2)`)
 *
 * @return Open flags
 */
[[nodiscard]] int open_flags(int fd);

/**
 * @brief Get file size from file descriptor `fstat(3)`
 *
 * @param file_descriptor Open file descriptor
 * @return The number of bytes
 */
[[nodiscard]] std::size_t get_file_size(std::string const& file_path);

/**
 * @brief Get file size given the file path
 *
 * @param file_path Path to a file
 * @return The number of bytes
 */
[[nodiscard]] std::size_t get_file_size(int file_descriptor);

/**
 * @brief Obtain the page cache residency information for a given file
 *
 * @param file_path Path to a file.
 * @return A pair containing the number of pages resident in the page cache and the total number of
 * pages.
 */
std::pair<std::size_t, std::size_t> get_page_cache_info(std::string const& file_path);

/**
 * @brief Obtain the page cache residency information for a given file
 *
 * @param fd File descriptor.
 * @return A pair containing the number of pages resident in the page cache and the total number of
 * pages.
 * @sa `get_page_cache_info(std::string const&)` overload.
 */
std::pair<std::size_t, std::size_t> get_page_cache_info(int fd);

/**
 * @brief Clear the page cache
 *
 * @param reclaim_dentries_and_inodes Whether to free reclaimable slab objects which include
 * dentries and inodes.
 * - If `true`, equivalent to executing `/sbin/sysctl vm.drop_caches=3`;
 * - If `false`, equivalent to executing `/sbin/sysctl vm.drop_caches=1`.
 * @param clear_dirty_pages Whether to trigger the writeback process to clear the dirty pages. If
 * `true`, `sync` will be called prior to cache clearing.
 * @return Whether the page cache has been successfully cleared
 *
 * @note This function creates a child process and executes the cache clearing shell command in the
 * following order
 * - Execute the command without `sudo` prefix. This is for the superuser and also for specially
 * configured systems where unprivileged users cannot execute `/usr/bin/sudo` but can execute
 * `/sbin/sysctl`. If this step succeeds, the function returns `true` immediately.
 * - Execute the command with `sudo` prefix. This is for the general case where selective
 * unprivileged users have permission to run `/sbin/sysctl` with `sudo` prefix.
 *
 * @exception kvikio::GenericSystemError if somehow the child process could not be created.
 */
bool clear_page_cache(bool reclaim_dentries_and_inodes = true, bool clear_dirty_pages = true);
}  // namespace kvikio
