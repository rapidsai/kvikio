/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <stdexcept>
#include <string>

#include <kvikio/shim/cufile_h_wrapper.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio {

/**
 * @brief Shim layer of the cuFile C-API
 *
 * This is a singleton class that use `dlopen` on construction to load the C-API of cuFile.
 *
 * For example, `cuFileAPI::instance().FileRead()` corresponds to calling `cuFileRead()`
 */
class cuFileAPI {
 public:
  decltype(cuFileHandleRegister)* HandleRegister{nullptr};
  decltype(cuFileHandleDeregister)* HandleDeregister{nullptr};
  decltype(cuFileRead)* Read{nullptr};
  decltype(cuFileWrite)* Write{nullptr};
  decltype(cuFileBufRegister)* BufRegister{nullptr};
  decltype(cuFileBufDeregister)* BufDeregister{nullptr};
  decltype(cuFileDriverGetProperties)* DriverGetProperties{nullptr};
  decltype(cuFileDriverSetPollMode)* DriverSetPollMode{nullptr};
  decltype(cuFileDriverSetMaxCacheSize)* DriverSetMaxCacheSize{nullptr};
  decltype(cuFileDriverSetMaxPinnedMemSize)* DriverSetMaxPinnedMemSize{nullptr};
  decltype(cuFileBatchIOSetUp)* BatchIOSetUp{nullptr};
  decltype(cuFileBatchIOSubmit)* BatchIOSubmit{nullptr};
  decltype(cuFileBatchIOGetStatus)* BatchIOGetStatus{nullptr};
  decltype(cuFileBatchIOCancel)* BatchIOCancel{nullptr};
  decltype(cuFileBatchIODestroy)* BatchIODestroy{nullptr};
  decltype(cuFileReadAsync)* ReadAsync{nullptr};
  decltype(cuFileWriteAsync)* WriteAsync{nullptr};
  decltype(cuFileStreamRegister)* StreamRegister{nullptr};
  decltype(cuFileStreamDeregister)* StreamDeregister{nullptr};

 private:
  // Don't call driver open and close directly, use `.driver_open()` and `.driver_close()`.
  decltype(cuFileDriverOpen)* DriverOpen{nullptr};
  decltype(cuFileDriverClose)* DriverClose{nullptr};

  // Don't call `GetVersion` directly, use `cuFileAPI::instance().version`.
  decltype(cuFileGetVersion)* GetVersion{nullptr};

 public:
  int version{0};

 private:
#ifdef KVIKIO_CUFILE_FOUND
  cuFileAPI()
  {
    // CUDA versions before CUDA 11.7.1 did not ship libcufile.so.0, so this is
    // a workaround that adds support for all prior versions of libcufile.
    void* lib = load_library({"libcufile.so.0",
                              "libcufile.so.1.3.0" /* 11.7.0 */,
                              "libcufile.so.1.2.1" /* 11.6.2, 11.6.1 */,
                              "libcufile.so.1.2.0" /* 11.6.0 */,
                              "libcufile.so.1.1.1" /* 11.5.1 */,
                              "libcufile.so.1.1.0" /* 11.5.0 */,
                              "libcufile.so.1.0.2" /* 11.4.4, 11.4.3, 11.4.2 */,
                              "libcufile.so.1.0.1" /* 11.4.1 */,
                              "libcufile.so.1.0.0" /* 11.4.0 */});
    get_symbol(HandleRegister, lib, KVIKIO_STRINGIFY(cuFileHandleRegister));
    get_symbol(HandleDeregister, lib, KVIKIO_STRINGIFY(cuFileHandleDeregister));
    get_symbol(Read, lib, KVIKIO_STRINGIFY(cuFileRead));
    get_symbol(Write, lib, KVIKIO_STRINGIFY(cuFileWrite));
    get_symbol(BufRegister, lib, KVIKIO_STRINGIFY(cuFileBufRegister));
    get_symbol(BufDeregister, lib, KVIKIO_STRINGIFY(cuFileBufDeregister));
    get_symbol(DriverOpen, lib, KVIKIO_STRINGIFY(cuFileDriverOpen));
    get_symbol(DriverClose, lib, KVIKIO_STRINGIFY(cuFileDriverClose));
    get_symbol(DriverGetProperties, lib, KVIKIO_STRINGIFY(cuFileDriverGetProperties));
    get_symbol(DriverSetPollMode, lib, KVIKIO_STRINGIFY(cuFileDriverSetPollMode));
    get_symbol(DriverSetMaxCacheSize, lib, KVIKIO_STRINGIFY(cuFileDriverSetMaxCacheSize));
    get_symbol(DriverSetMaxPinnedMemSize, lib, KVIKIO_STRINGIFY(cuFileDriverSetMaxPinnedMemSize));

#ifdef KVIKIO_CUFILE_VERSION_API_FOUND
    try {
      get_symbol(GetVersion, lib, KVIKIO_STRINGIFY(cuFileGetVersion));
      int ver;
      CUfileError_t const error = GetVersion(&ver);
      if (error.err == CU_FILE_SUCCESS) { version = ver; }
    } catch (std::runtime_error const&) {
    }
#endif

    if (version >= 1060) {
      get_symbol(BatchIOSetUp, lib, KVIKIO_STRINGIFY(cuFileBatchIOSetUp));
      get_symbol(BatchIOSubmit, lib, KVIKIO_STRINGIFY(cuFileBatchIOSubmit));
      get_symbol(BatchIOGetStatus, lib, KVIKIO_STRINGIFY(cuFileBatchIOGetStatus));
      get_symbol(BatchIOCancel, lib, KVIKIO_STRINGIFY(cuFileBatchIOCancel));
      get_symbol(BatchIODestroy, lib, KVIKIO_STRINGIFY(cuFileBatchIODestroy));
    }

    if (version >= 1070) {
      get_symbol(ReadAsync, lib, KVIKIO_STRINGIFY(cuFileReadAsync));
      get_symbol(WriteAsync, lib, KVIKIO_STRINGIFY(cuFileWriteAsync));
      get_symbol(StreamRegister, lib, KVIKIO_STRINGIFY(cuFileStreamRegister));
      get_symbol(StreamDeregister, lib, KVIKIO_STRINGIFY(cuFileStreamDeregister));
    }

    // cuFile is supposed to open and close the driver automatically but
    // because of a bug in cuFile v1.4 (CUDA v11.8) it sometimes segfaults:
    // <https://github.com/rapidsai/kvikio/issues/159>.
    if (version < 1050) { driver_open(); }
  }

  // Notice, we have to close the driver at program exit (if we opened it) even though we are
  // not allowed to call CUDA after main[1]. This is because, cuFile will segfault if the
  // driver isn't closed on program exit i.e. we are doomed if we do, doomed if we don't, but
  // this seems to be the lesser of two evils.
  // [1] <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
  ~cuFileAPI()
  {
    if (version < 1050) { driver_close(); }
  }
#else
  cuFileAPI() { throw std::runtime_error("KvikIO not compiled with cuFile.h"); }
#endif

 public:
  cuFileAPI(cuFileAPI const&)       = delete;
  void operator=(cuFileAPI const&)  = delete;
  cuFileAPI(cuFileAPI const&&)      = delete;
  void operator=(cuFileAPI const&&) = delete;

  KVIKIO_EXPORT static cuFileAPI& instance()
  {
    static cuFileAPI _instance;
    return _instance;
  }

  /**
   * @brief Open the cuFile driver
   *
   * cuFile allows multiple calls to `cufileDriverOpen()`, only the first call opens
   * the driver, but every call should have a matching call to `cufileDriverClose()`.
   */
  void driver_open()
  {
    CUfileError_t const error = DriverOpen();
    if (error.err != CU_FILE_SUCCESS) {
      throw std::runtime_error(std::string{"Unable to open GDS file driver: "} +
                               cufileop_status_error(error.err));
    }
  }

  /**
   * @brief Close the cuFile driver
   */
  void driver_close()
  {
    CUfileError_t const error = DriverClose();
    if (error.err != CU_FILE_SUCCESS) {
      throw std::runtime_error(std::string{"Unable to close GDS file driver: "} +
                               cufileop_status_error(error.err));
    }
  }
};

/**
 * @brief Check if the cuFile library is available
 *
 * Notice, this doesn't check if the runtime environment supports cuFile.
 *
 * @return The boolean answer
 */
#ifdef KVIKIO_CUFILE_FOUND
inline bool is_cufile_library_available()
{
  try {
    cuFileAPI::instance();
  } catch (const std::runtime_error&) {
    return false;
  }
  return true;
}
#else
constexpr bool is_cufile_library_available() { return false; }
#endif

/**
 * @brief Check if the cuFile is available and expected to work
 *
 * Besides checking if the cuFile library is available, this also checks the
 * runtime environment.
 *
 * @return The boolean answer
 */
inline bool is_cufile_available()
{
  return is_cufile_library_available() && run_udev_readable() && !is_running_in_wsl();
}

/**
 * @brief Get cufile version (or zero if older than v1.8).
 *
 * The version is returned as (1000 major + 10 minor). E.g., cufile v1.8.0 would
 * be represented by 1080.
 *
 * Notice, this is not the version of the CUDA toolkit. cufile is part of the
 * toolkit but follows its own version scheme.
 *
 * @return The version (1000 major + 10 minor) or zero if older than 1080.
 */
#ifdef KVIKIO_CUFILE_FOUND
inline int cufile_version()
{
  try {
    return cuFileAPI::instance().version;
  } catch (std::runtime_error const&) {
    return 0;
  }
}
#else
constexpr bool cufile_version() { return 0; }
#endif

/**
 * @brief Check if cuFile's batch API is available.
 *
 * Since `cuFileGetVersion()` first became available in cufile v1.8 (CTK v12.3),
 * this function returns false for versions older than v1.8 even though the batch
 * API became available in v1.6.
 *
 * @return The boolean answer
 */
inline bool is_batch_api_available() noexcept { return cufile_version() > 1060; }

/**
 * @brief Check if cuFile's stream (async) API is available.
 *
 * Since `cuFileGetVersion()` first became available in cufile v1.8 (CTK v12.3),
 * this function returns false for versions older than v1.8 even though the stream
 * API became available in v1.7.
 *
 * @return The boolean answer
 */
inline bool is_stream_api_available() noexcept { return cufile_version() > 1070; }

}  // namespace kvikio
