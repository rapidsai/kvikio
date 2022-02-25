/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#pragma once
#include <kvikio/error.hpp>

#include <cufile.h>
#include <dlfcn.h>

namespace kvikio {
namespace {
template <typename T>
void get_symbol(T& handle, void* lib, const char* name)
{
  /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)*/
  handle = reinterpret_cast<T>(dlsym(lib, name));
  if (handle == nullptr) {
    throw CUfileException{std::string{"Cannot load symbol: "} + __FILE__ + ":" +
                          CUFILE_STRINGIFY(__LINE__) + ": " + name};
  }
}

}  // namespace

class CAPI {
 public:
  decltype(cuFileHandleRegister)* HandleRegister{nullptr};
  decltype(cuFileHandleDeregister)* HandleDeregister{nullptr};
  decltype(cuFileRead)* Read{nullptr};
  decltype(cuFileWrite)* Write{nullptr};
  decltype(cuFileBufRegister)* BufRegister{nullptr};
  decltype(cuFileBufDeregister)* BufDeregister{nullptr};
  decltype(cuFileDriverOpen)* DriverOpen{nullptr};
  decltype(cuFileDriverClose)* DriverClose{nullptr};
  decltype(cuFileDriverGetProperties)* DriverGetProperties{nullptr};
  decltype(cuFileDriverSetPollMode)* DriverSetPollMode{nullptr};
  decltype(cuFileDriverSetMaxCacheSize)* DriverSetMaxCacheSize{nullptr};
  decltype(cuFileDriverSetMaxPinnedMemSize)* DriverSetMaxPinnedMemSize{nullptr};

  CAPI()
  {
    dlerror();  // Clear old errors
    void* lib       = dlopen("libcufile.so", RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
    const char* err = dlerror();
    if (err != nullptr) {
      throw CUfileException{std::string{"Cannot load `libcufile.so`: "} + __FILE__ + ":" +
                            CUFILE_STRINGIFY(__LINE__) + ": " + err};
    }
    get_symbol(HandleRegister, lib, "cuFileHandleRegister");
    get_symbol(HandleDeregister, lib, "cuFileHandleDeregister");
    get_symbol(Read, lib, "cuFileRead");
    get_symbol(Write, lib, "cuFileWrite");
    get_symbol(BufRegister, lib, "cuFileBufRegister");
    get_symbol(BufDeregister, lib, "cuFileBufDeregister");
    get_symbol(DriverOpen, lib, "cuFileDriverOpen");
    get_symbol(DriverClose, lib, "cuFileDriverClose");
    get_symbol(DriverGetProperties, lib, "cuFileDriverGetProperties");
    get_symbol(DriverSetPollMode, lib, "cuFileDriverSetPollMode");
    get_symbol(DriverSetMaxCacheSize, lib, "cuFileDriverSetMaxCacheSize");
    get_symbol(DriverSetMaxPinnedMemSize, lib, "cuFileDriverSetMaxPinnedMemSize");
  }

  static CAPI* instance()
  {
    static CAPI _instance;
    return &_instance;
  }
};

}  // namespace kvikio
