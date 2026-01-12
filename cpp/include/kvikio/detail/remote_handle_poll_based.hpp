/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

class RemoteHandlePollBased {
 public:
  RemoteHandlePollBased(std::string const& url);

  std::size_t pread(void* buf, std::size_t size, std::size_t file_offset = 0);

 private:
  CURLM* _multi;
  std::string _url;
};
}  // namespace kvikio::detail
