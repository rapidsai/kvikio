/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

struct curl_slist;

namespace kvikio::detail {

/**
 * @brief Free a libcurl header list with `curl_slist_free_all()`.
 */
struct CurlHeaderListDeleter {
  void operator()(curl_slist* list) const noexcept;
};

/**
 * @brief A libcurl header list with lifetime management.
 */
using CurlHeaderList = std::unique_ptr<curl_slist, CurlHeaderListDeleter>;

/**
 * @brief Build a libcurl header list.
 *
 * @param headers The headers, each as a `"<name>: <value>"` string.
 * @return The list, or an empty `CurlHeaderList` if `headers` is empty.
 * @exception std::runtime_error if `curl_slist_append()` fails.
 */
CurlHeaderList make_curl_header_list(std::vector<std::string> const& headers);

}  // namespace kvikio::detail
