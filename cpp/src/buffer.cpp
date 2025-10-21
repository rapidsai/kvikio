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

#include <algorithm>
#include <iostream>
#include <vector>

#include <kvikio/buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

void buffer_register(void const* devPtr_base,
                     std::size_t size,
                     int flags,
                     std::vector<int> const& errors_to_ignore)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (defaults::is_compat_mode_preferred()) { return; }
  CUfileError_t status = cuFileAPI::instance().BufRegister(devPtr_base, size, flags);
  if (status.err != CU_FILE_SUCCESS) {
    // Check if `status.err` is in `errors_to_ignore`
    if (std::find(errors_to_ignore.begin(), errors_to_ignore.end(), status.err) ==
        errors_to_ignore.end()) {
      CUFILE_TRY(status);
    }
  }
}

void buffer_deregister(void const* devPtr_base)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (defaults::is_compat_mode_preferred()) { return; }
  CUFILE_TRY(cuFileAPI::instance().BufDeregister(devPtr_base));
}

void memory_register(void const* devPtr, int flags, std::vector<int> const& errors_to_ignore)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto [base, nbytes, offset] = get_alloc_info(devPtr);
  buffer_register(base, nbytes, flags, errors_to_ignore);
}

void memory_deregister(void const* devPtr)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto [base, nbytes, offset] = get_alloc_info(devPtr);
  buffer_deregister(base);
}

}  // namespace kvikio
