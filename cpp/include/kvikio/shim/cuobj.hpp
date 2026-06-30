/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <string>

#include <kvikio/shim/utils.hpp>

namespace kvikio {

/**
 * @brief Shim layer over the NVIDIA cuObject C-ABI used for S3-over-RDMA.
 *
 * Singleton that `dlopen`s ``libkvikio_cuobj_shim.so`` on construction and
 * resolves its C entry points. The shim wraps the cuObject ``cuObjClient`` C++
 * class (see ``cpp/cuobj_shim/cuobj_shim.cpp``); loading it at runtime keeps
 * cuObject an optional dependency, mirroring how :class:`cuFileAPI` loads
 * ``libcufile.so.0``. The library path can be overridden with the
 * ``KVIKIO_CUOBJ_SHIM`` environment variable.
 */
class cuObjAPI {
 public:
  int (*Available)(){nullptr};
  int (*RegisterBuffer)(void*, std::size_t){nullptr};
  int (*DeregisterBuffer)(void*){nullptr};
  char const* (*GetRDMAToken)(void*, std::size_t, std::size_t, int){nullptr};
  int (*PutRDMAToken)(char const*){nullptr};

 private:
  cuObjAPI();

 public:
  cuObjAPI(cuObjAPI const&)            = delete;
  cuObjAPI& operator=(cuObjAPI const&) = delete;

  KVIKIO_EXPORT static cuObjAPI& instance();
};

/**
 * @brief Whether cuObject S3-over-RDMA is usable.
 *
 * ``true`` only when the shim library loads and a cuObject client connection
 * can be established (RDMA-capable NIC and a reachable RDMA S3 endpoint).
 */
[[nodiscard]] bool is_cuobj_available() noexcept;

}  // namespace kvikio
