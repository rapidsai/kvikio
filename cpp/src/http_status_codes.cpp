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

#include <cstddef>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <kvikio/error.hpp>

namespace kvikio {

namespace detail {
std::vector<int> parse_http_status_codes(std::string_view env_var_name,
                                         std::string const& status_codes)
{
  // Ensure `status_codes` consists only of 3-digit integers separated by commas, allowing spaces.
  std::regex const check_pattern(R"(^\s*\d{3}\s*(\s*,\s*\d{3}\s*)*$)");
  KVIKIO_EXPECT(std::regex_match(status_codes, check_pattern),
                std::string{env_var_name} + ": invalid format, expected comma-separated integers.",
                std::invalid_argument);

  // Match every integer in `status_codes`.
  std::regex const number_pattern(R"(\d+)");

  // For each match, we push_back `std::stoi(match.str())` into `ret`.
  std::vector<int> ret;
  std::transform(std::sregex_iterator(status_codes.begin(), status_codes.end(), number_pattern),
                 std::sregex_iterator(),
                 std::back_inserter(ret),
                 [](std::smatch const& match) -> int { return std::stoi(match.str()); });
  return ret;
}

}  // namespace detail

}  // namespace kvikio
