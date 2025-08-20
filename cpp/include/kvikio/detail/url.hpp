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

#include <optional>

#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {
/**
 * @brief RAII URL handle. Not copy-constructible or copy-assignable, but move-constructible and
 * move-assignable.
 */
class CurlUrlHandle {
 private:
  CURLU* _handle{nullptr};

 public:
  CurlUrlHandle();
  ~CurlUrlHandle();

  CurlUrlHandle(CurlUrlHandle const&)            = delete;
  CurlUrlHandle& operator=(CurlUrlHandle const&) = delete;

  CurlUrlHandle(CurlUrlHandle&& other) noexcept;
  CurlUrlHandle& operator=(CurlUrlHandle&& other) noexcept;

  CURLU* get() const;
};

class UrlParser {
 private:
  static std::optional<std::string> extract_component(
    CurlUrlHandle& handle,
    CURLUPart part,
    unsigned int bitmask_component_flags,
    std::optional<CURLUcode> exempt_err_code = std::nullopt);

 public:
  struct UrlComponents {
    std::optional<std::string> scheme;
    std::optional<std::string> host;
    std::optional<std::string> port;
    std::optional<std::string> path;
    std::optional<std::string> query;
    std::optional<std::string> fragment;
  };

  static UrlComponents parse(std::string const& url,
                             std::optional<unsigned int> bitmask_url_flags       = std::nullopt,
                             std::optional<unsigned int> bitmask_component_flags = std::nullopt);
};
}  // namespace kvikio::detail
