/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <curl/curl.h>

#include <kvikio/detail/curl_header_list.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

void CurlHeaderListDeleter::operator()(curl_slist* list) const noexcept
{
  curl_slist_free_all(list);
}

CurlHeaderList make_curl_header_list(std::vector<std::string> const& headers)
{
  CurlHeaderList list;
  for (auto const& header : headers) {
    auto* const new_list = curl_slist_append(list.get(), header.c_str());
    KVIKIO_EXPECT(new_list != nullptr, "curl_slist_append() failed", std::runtime_error);
    // According to libcurl's doc, we should use the returned pointer in place of the old list
    // pointer.
    std::ignore = list.release();
    list.reset(new_list);
  }
  return list;
}

}  // namespace kvikio::detail
