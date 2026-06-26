/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <kvikio/shim/cuobj.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio {

namespace {
std::string cuobj_shim_library_name()
{
  if (char const* env = std::getenv("KVIKIO_CUOBJ_SHIM"); env != nullptr && *env != '\0') {
    return std::string{env};
  }
  return "libkvikio_cuobj_shim.so";
}
}  // namespace

cuObjAPI::cuObjAPI()
{
  void* lib = load_library(cuobj_shim_library_name());
  get_symbol(Available, lib, "kvikio_cuobj_available");
  get_symbol(RegisterBuffer, lib, "kvikio_cuobj_register_buffer");
  get_symbol(DeregisterBuffer, lib, "kvikio_cuobj_deregister_buffer");
  get_symbol(GetRDMAToken, lib, "kvikio_cuobj_get_rdma_token");
  get_symbol(PutRDMAToken, lib, "kvikio_cuobj_put_rdma_token");
}

cuObjAPI& cuObjAPI::instance()
{
  static cuObjAPI _instance;
  return _instance;
}

bool is_cuobj_available() noexcept
{
  try {
    return cuObjAPI::instance().Available() != 0;
  } catch (std::runtime_error const&) {
    return false;
  }
}

}  // namespace kvikio
