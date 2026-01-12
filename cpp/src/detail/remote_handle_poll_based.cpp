/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/remote_handle_poll_based.hpp>

namespace kvikio::detail {
RemoteHandlePollBased::RemoteHandlePollBased(std::string const& url) {}

std::size_t RemoteHandlePollBased::pread(void* buf, std::size_t size, std::size_t file_offset)
{
  return 123;
}
}  // namespace kvikio::detail
