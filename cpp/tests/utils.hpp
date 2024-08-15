/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <filesystem>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

namespace kvikio::test {

class TempDir {
 public:
  TempDir(const bool cleanup = true) : _cleanup{cleanup}
  {
    std::string tpl{std::filesystem::temp_directory_path() / "legate-dataframe.XXXXXX"};
    if (mkdtemp(tpl.data()) == nullptr) {}
    _dir_path = tpl;
  }
  ~TempDir() noexcept
  {
    if (_cleanup) {
      try {
        std::filesystem::remove_all(_dir_path);
      } catch (...) {
        std::cout << "error while trying to remove " << _dir_path.string() << std::endl;
      }
    }
  }

  const std::filesystem::path& path() { return _dir_path; }

  operator std::string() { return path(); }

 private:
  const bool _cleanup;
  std::filesystem::path _dir_path;
};

}  // namespace kvikio::test
