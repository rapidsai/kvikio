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

#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>

namespace kvikio::test {
/**
 * @brief RAII class used to temporarily set environment variables to new values upon construction,
 * and restore them to previous values upon destruction.
 */
class EnvVarContext {
 private:
  /**
   * @brief The state of an environment variable
   */
  struct EnvVarState {
    // Whether the environment variable existed before entering the context
    bool existed_before{};
    // The previous value of the environment variable, if existed
    std::string previous_value{};
  };

 public:
  /**
   * @brief Set the environment variables to new values
   *
   * @param env_var_entries User-specified environment variables. Each entry includes the variable
   * name and value.
   */
  EnvVarContext(
    std::initializer_list<std::pair<std::string_view, std::string_view>> env_var_entries);

  /**
   * @brief Restore the environment variables to previous values
   *
   * Reset the environment variables to their previous states:
   * - If one existed before, restore to its previous value.
   * - Otherwise, remove it from the environment.
   */
  ~EnvVarContext();

  EnvVarContext(EnvVarContext const&)            = delete;
  EnvVarContext(EnvVarContext&&)                 = delete;
  EnvVarContext& operator=(EnvVarContext const&) = delete;
  EnvVarContext& operator=(EnvVarContext&&)      = delete;

 private:
  std::unordered_map<std::string, EnvVarState> _env_var_map;
};
}  // namespace kvikio::test
