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

#include <sys/syscall.h>
#include <array>
#include <sstream>

#ifdef KVIKIO_CUDA_FOUND
#include <nvtx3/nvtx3.hpp>
#endif

#include <kvikio/nvtx.hpp>

namespace kvikio {

nvtx_manager& nvtx_manager::instance() noexcept
{
  static nvtx_manager _instance;
  return _instance;
}

const nvtx_color_type& nvtx_manager::default_color() noexcept
{
#ifdef KVIKIO_CUDA_FOUND
  static nvtx_color_type default_color{nvtx3::argb{0, 255, 255, 255}};
  return default_color;
#else
  static nvtx_color_type dummy{};
  return dummy;
#endif
}

const nvtx_color_type& nvtx_manager::get_color_by_index(std::uint64_t idx) noexcept
{
#ifdef KVIKIO_CUDA_FOUND
  constexpr std::size_t num_color{16};
  static_assert((num_color & (num_color - 1)) == 0);  // Is power of 2
  static std::array<nvtx_color_type, num_color> color_palette = {nvtx3::rgb{106, 192, 67},
                                                                 nvtx3::rgb{191, 73, 203},
                                                                 nvtx3::rgb{93, 151, 76},
                                                                 nvtx3::rgb{96, 72, 194},
                                                                 nvtx3::rgb{179, 170, 71},
                                                                 nvtx3::rgb{92, 58, 113},
                                                                 nvtx3::rgb{212, 136, 57},
                                                                 nvtx3::rgb{96, 144, 194},
                                                                 nvtx3::rgb{211, 69, 56},
                                                                 nvtx3::rgb{97, 179, 155},
                                                                 nvtx3::rgb{203, 69, 131},
                                                                 nvtx3::rgb{57, 89, 48},
                                                                 nvtx3::rgb{184, 133, 199},
                                                                 nvtx3::rgb{128, 102, 51},
                                                                 nvtx3::rgb{211, 138, 130},
                                                                 nvtx3::rgb{122, 50, 49}};
  auto safe_idx = idx & (num_color - 1);  // idx % num_color
  return color_palette[safe_idx];
#else
  static nvtx_color_type dummy{};
  return dummy;
#endif
}

void nvtx_manager::rename_current_thread(std::string_view new_name) noexcept
{
#ifdef KVIKIO_CUDA_FOUND
  auto tid = syscall(SYS_gettid);
  std::stringstream ss;
  ss << new_name << " (" << tid << ")";

  nvtxResourceAttributes_t attribs = {0};
  attribs.version                  = NVTX_VERSION;
  attribs.size                     = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
  attribs.identifierType           = NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE;
  attribs.identifier.ullValue      = tid;
  attribs.messageType              = NVTX_MESSAGE_TYPE_ASCII;
  attribs.message.ascii            = ss.str().c_str();
  nvtxResourceHandle_t handle =
    nvtxDomainResourceCreate(nvtx3::domain::get<libkvikio_domain>(), &attribs);
#endif
}

}  // namespace kvikio
