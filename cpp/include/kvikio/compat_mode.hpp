/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
