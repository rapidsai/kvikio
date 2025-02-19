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

#include <cstdint>
#include <string>

#include <kvikio/file_utils.hpp>

namespace kvikio {
/**
 * @brief I/O compatibility mode.
 */
enum class CompatMode : uint8_t {
  OFF,  ///< Enforce cuFile I/O. GDS will be activated if the system requirements for cuFile are met
        ///< and cuFile is properly configured. However, if the system is not suited for cuFile, I/O
        ///< operations under the OFF option may error out.
  ON,   ///< Enforce POSIX I/O.
  AUTO,  ///< Try cuFile I/O first, and fall back to POSIX I/O if the system requirements for cuFile
         ///< are not met.
};

namespace detail {
/**
 * @brief Parse a string into a CompatMode enum.
 *
 * @param compat_mode_str Compatibility mode in string format (case-insensitive). Valid values
 * are:
 *   - `ON` (alias: `TRUE`, `YES`, `1`)
 *   - `OFF` (alias: `FALSE`, `NO`, `0`)
 *   - `AUTO`
 * @return A CompatMode enum.
 */
CompatMode parse_compat_mode_str(std::string_view compat_mode_str);

}  // namespace detail

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
   * @brief Functionally identical to defaults::infer_compat_mode_if_auto(CompatMode).
   *
   * @param compat_mode Compatibility mode.
   * @return If the given compatibility mode is CompatMode::AUTO, infer the final compatibility
   * mode.
   */
  CompatMode infer_compat_mode_if_auto(CompatMode compat_mode) noexcept;

  /**
   * @brief Functionally identical to defaults::is_compat_mode_preferred(CompatMode).
   *
   * @param compat_mode Compatibility mode.
   * @return Boolean answer.
   */
  bool is_compat_mode_preferred(CompatMode compat_mode) noexcept;

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
