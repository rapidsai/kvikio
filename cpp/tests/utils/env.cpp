/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "env.hpp"
#include <cstdlib>
#include <sstream>
#include <utility>

#include <kvikio/error.hpp>

namespace kvikio::test {
EnvVarContext::EnvVarContext(
  std::initializer_list<std::pair<std::string_view, std::string_view>> env_var_entries)
{
  for (auto const& [key, current_value] : env_var_entries) {
    EnvVarState env_var_state;
    if (auto const res = std::getenv(key.data()); res != nullptr) {
      env_var_state.existed_before = true;
      env_var_state.previous_value = res;
    }
    SYSCALL_CHECK(setenv(key.data(), current_value.data(), 1 /* allow overwrite */));
    std::string key_str{key};
    if (_env_var_map.find(key_str) != _env_var_map.end()) {
      std::stringstream ss;
      ss << "Environment variable " << key << " has already been set in this context.";
      KVIKIO_FAIL(ss.str());
    }
    _env_var_map.insert({std::move(key_str), std::move(env_var_state)});
  }
}

EnvVarContext::~EnvVarContext()
{
  for (auto const& [key, state] : _env_var_map) {
    if (state.existed_before) {
      SYSCALL_CHECK(setenv(key.c_str(), state.previous_value.c_str(), 1 /* allow overwrite */));
    } else {
      SYSCALL_CHECK(unsetenv(key.c_str()));
    }
  }
}
}  // namespace kvikio::test
