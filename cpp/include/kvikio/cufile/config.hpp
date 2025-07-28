/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
