/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/remote_handle_poll_based.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

RemoteHandlePollBased::RemoteHandlePollBased(std::string const& url) : _url{url}
{
  _multi = curl_multi_init();
  KVIKIO_EXPECT(_multi != nullptr, "Failed to initialize libcurl multi API");
}

RemoteHandlePollBased::~RemoteHandlePollBased()
{
  try {
    KVIKIO_CHECK_CURL_MULTI(curl_multi_cleanup(_multi));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(e.what());
  }
}

std::size_t RemoteHandlePollBased::pread(void* buf, std::size_t size, std::size_t file_offset)
{
  return 123;
}
}  // namespace kvikio::detail
