/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief Get the filepath to cuFile's config file (`cufile.json`) or the empty string
 *
 * This lookup is cached.
 *
 * @return The filepath to the cufile.json file or the empty string if it isn't found.
 */
[[nodiscard]] KVIKIO_EXPORT std::string const& config_path();

}  // namespace kvikio
