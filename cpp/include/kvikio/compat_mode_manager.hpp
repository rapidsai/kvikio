/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <kvikio/compat_mode.hpp>

namespace kvikio {

// Forward declaration.
class FileHandle;

/**
 * @brief Store and manage the compatibility mode data associated with a FileHandle.
 */
class CompatModeManager {
 private:
  CompatMode _compat_mode_requested{CompatMode::AUTO};
  bool _is_compat_mode_preferred{true};
  bool _is_compat_mode_preferred_for_async{true};

 public:
  /**
   * @brief Construct an empty compatibility mode manager.
   */
  CompatModeManager() noexcept = default;

  /**
   * @brief Construct a compatibility mode manager associated with a FileHandle.
   *
   * According to the file path, requested compatibility mode, and the system configuration, the
   * compatibility manager:
   * - Infers the final compatibility modes for synchronous and asynchronous I/O paths,
   * respectively.
   * - Initializes the file wrappers and cuFile handle associated with a FileHandle.
   *
   * @param file_path Refer to
   * FileHandle::FileHandle(std::string const&, std::string const&, mode_t, CompatMode).
   * @param flags Same as above.
   * @param mode Same as above.
   * @param compat_mode_requested Same as above.
   * @param file_handle Pointer to the FileHandle object that owns this compatibility mode manager.
   */
  CompatModeManager(std::string const& file_path,
                    std::string const& flags,
                    mode_t mode,
                    CompatMode compat_mode_requested,
                    FileHandle* file_handle);

  ~CompatModeManager() noexcept                              = default;
  CompatModeManager(const CompatModeManager&)                = default;
  CompatModeManager& operator=(const CompatModeManager&)     = default;
  CompatModeManager(CompatModeManager&&) noexcept            = default;
  CompatModeManager& operator=(CompatModeManager&&) noexcept = default;

  /**
   * @brief Check if the compatibility mode for synchronous I/O of the associated FileHandle is
   * expected to be CompatMode::ON.
   *
   * @return Boolean answer.
   */
  bool is_compat_mode_preferred() const noexcept;

  /**
   * @brief Check if the compatibility mode for asynchronous I/O of the associated FileHandle is
   * expected to be CompatMode::ON.
   *
   * @return Boolean answer.
   */
  bool is_compat_mode_preferred_for_async() const noexcept;

  /**
   * @brief Retrieve the original compatibility mode requested.
   *
   * @return The original compatibility mode requested.
   */
  CompatMode compat_mode_requested() const noexcept;

  /**
   * @brief Determine if asynchronous I/O can be performed or not (throw exceptions)
   * according to the existing compatibility mode data in the manager.
   *
   * Asynchronous I/O cannot be performed, for instance, when compat_mode_requested() is
   * CompatMode::OFF, is_compat_mode_preferred() is CompatMode::OFF, but
   * is_compat_mode_preferred_for_async() is CompatMode::ON (due to missing cuFile stream API or
   * cuFile configuration file).
   */
  void validate_compat_mode_for_async() const;
};

}  // namespace kvikio
