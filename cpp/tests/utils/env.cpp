/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "env.hpp"
#include <cstdlib>
#include <sstream>
#include <utility>

#include <kvikio/error.hpp>

namespace kvikio::test {

EnvVarContext::EnvVarContext(
  std::initializer_list<std::pair<std::string_view, std::string_view>> entries)
{
  for (auto const& [key, value] : entries) {
    add_entry(key, value);
  }
}

EnvVarContext::EnvVarContext(std::unordered_map<std::string, std::string> const& entries)
{
  for (auto const& [key, value] : entries) {
    add_entry(key, value);
  }
}

EnvVarContext::~EnvVarContext()
{
  for (auto const& [key, state] : _env_var_map) {
    if (state.existed_before) {
      KVIKIO_SYSCALL_CHECK(
        setenv(key.c_str(), state.previous_value.c_str(), 1 /* allow overwrite */));
    } else {
      KVIKIO_SYSCALL_CHECK(unsetenv(key.c_str()));
    }
  }
}

void EnvVarContext::add_entry(std::string_view key, std::string_view value)
{
  std::string key_str{key};
  if (_env_var_map.contains(key_str)) {
    KVIKIO_FAIL("Environment variable " + key_str + " has already been set in this context.");
  }
  EnvVarState state;
  if (auto const res = std::getenv(key.data()); res != nullptr) {
    state.existed_before = true;
    state.previous_value = res;
  }
  KVIKIO_SYSCALL_CHECK(setenv(key.data(), value.data(), 1 /* allow overwrite */));
  _env_var_map.insert({std::move(key_str), std::move(state)});
}
}  // namespace kvikio::test
