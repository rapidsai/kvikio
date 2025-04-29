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
 * @throw std::invalid_argument if the specified flags are not supported.
 * @throw std::invalid_argument if `o_direct` is true, but `O_DIRECT` is not supported.
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
}  // namespace kvikio
