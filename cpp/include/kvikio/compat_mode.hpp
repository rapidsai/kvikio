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
#include <string_view>

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

}  // namespace kvikio
