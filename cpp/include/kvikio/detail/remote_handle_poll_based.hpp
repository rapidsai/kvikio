/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include <kvikio/shim/libcurl.hpp>

#define KVIKIO_CHECK_CURL_MULTI(err_code) \
  kvikio::detail::check_curl_multi(err_code, __FILE__, __LINE__)

namespace kvikio::detail {

inline void check_curl_multi(CURLMcode err_code, char const* filename, int line_number)
{
  if (err_code == CURLMcode::CURLM_OK) { return; }
  std::stringstream ss;
  ss << "libcurl error: " << curl_multi_strerror(err_code) << " at: " << filename << ":"
     << line_number << "\n";
  throw std::runtime_error(ss.str());
}

class RemoteHandlePollBased {
 public:
  RemoteHandlePollBased(std::string const& url);

  ~RemoteHandlePollBased();

  std::size_t pread(void* buf, std::size_t size, std::size_t file_offset = 0);

 private:
  CURLM* _multi;
  std::string _url;
  std::size_t _num_conns{8};
};
}  // namespace kvikio::detail
