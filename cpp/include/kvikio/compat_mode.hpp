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
 * include:
 *   - `ON` (alias: `TRUE`, `YES`, `1`)
 *   - `OFF` (alias: `FALSE`, `NO`, `0`)
 *   - `AUTO`
 * @return A CompatMode enum.
 */
CompatMode parse_compat_mode_str(std::string_view compat_mode_str);

}  // namespace detail

/**
 * @brief
 *
 */
class CompatModeManager {
 private:
  CompatMode _compat_mode_requested{CompatMode::AUTO};
  bool _is_compat_mode_preferred{true};
  bool _is_compat_mode_preferred_for_async{true};

 public:
  CompatModeManager() noexcept = default;
  CompatModeManager(std::string const& file_path,
                    std::string const& flags,
                    mode_t mode,
                    CompatMode compat_mode_requested,
                    FileWrapper& file_direct_on,
                    FileWrapper& file_direct_off,
                    CUFileHandleWrapper& cufile_handle);
  ~CompatModeManager() noexcept                              = default;
  CompatModeManager(const CompatModeManager&)                = default;
  CompatModeManager& operator=(const CompatModeManager&)     = default;
  CompatModeManager(CompatModeManager&&) noexcept            = default;
  CompatModeManager& operator=(CompatModeManager&&) noexcept = default;

  void compat_mode_reset(CompatMode compat_mode_requested);

  CompatMode infer_compat_mode_if_auto(CompatMode compat_mode) noexcept;

  bool is_compat_mode_preferred(CompatMode compat_mode) noexcept;

  bool is_compat_mode_preferred() const noexcept;

  bool is_compat_mode_preferred_for_async() const noexcept;

  CompatMode compat_mode_requested() const noexcept;

  /**
   * @brief Determine if the asynchronous I/O should be performed or not (throw exceptions)
   * according to `_compat_mode_requested`, `_is_compat_mode_preferred`, and
   * `_is_compat_mode_preferred_for_async`.
   */
  void validate_compat_mode_for_async();
};

}  // namespace kvikio
