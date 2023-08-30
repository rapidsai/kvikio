/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <iostream>
#include <kvikio/shim/cufile_h_wrapper.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio {

#ifdef KVIKIO_CUFILE_FOUND

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
  decltype(cuFileDriverOpen)* DriverOpen{nullptr};
  decltype(cuFileDriverClose)* DriverClose{nullptr};
  decltype(cuFileDriverGetProperties)* DriverGetProperties{nullptr};
  decltype(cuFileDriverSetPollMode)* DriverSetPollMode{nullptr};
  decltype(cuFileDriverSetMaxCacheSize)* DriverSetMaxCacheSize{nullptr};
  decltype(cuFileDriverSetMaxPinnedMemSize)* DriverSetMaxPinnedMemSize{nullptr};

#ifdef KVIKIO_CUFILE_BATCH_API_FOUND
  decltype(cuFileBatchIOSetUp)* BatchIOSetUp{nullptr};
  decltype(cuFileBatchIOSubmit)* BatchIOSubmit{nullptr};
  decltype(cuFileBatchIOGetStatus)* BatchIOGetStatus{nullptr};
  decltype(cuFileBatchIOCancel)* BatchIOCancel{nullptr};
  decltype(cuFileBatchIODestroy)* BatchIODestroy{nullptr};
#endif

#ifdef KVIKIO_CUFILE_STREAM_API_FOUND
  decltype(cuFileReadAsync)* ReadAsync{nullptr};
  decltype(cuFileWriteAsync)* WriteAsync{nullptr};
  decltype(cuFileStreamRegister)* StreamRegister{nullptr};
  decltype(cuFileStreamDeregister)* StreamDeregister{nullptr};
#endif
  bool stream_available = false;

 private:
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

#ifdef KVIKIO_CUFILE_BATCH_API_FOUND
    get_symbol(BatchIOSetUp, lib, KVIKIO_STRINGIFY(cuFileBatchIOSetUp));
    get_symbol(BatchIOSubmit, lib, KVIKIO_STRINGIFY(cuFileBatchIOSubmit));
    get_symbol(BatchIOGetStatus, lib, KVIKIO_STRINGIFY(cuFileBatchIOGetStatus));
    get_symbol(BatchIOCancel, lib, KVIKIO_STRINGIFY(cuFileBatchIOCancel));
    get_symbol(BatchIODestroy, lib, KVIKIO_STRINGIFY(cuFileBatchIODestroy));
#endif

#ifdef KVIKIO_CUFILE_STREAM_API_FOUND
    get_symbol(ReadAsync, lib, KVIKIO_STRINGIFY(cuFileReadAsync));
    get_symbol(WriteAsync, lib, KVIKIO_STRINGIFY(cuFileWriteAsync));
    get_symbol(StreamRegister, lib, KVIKIO_STRINGIFY(cuFileStreamRegister));
    get_symbol(StreamDeregister, lib, KVIKIO_STRINGIFY(cuFileStreamDeregister));
    try {
      void* s{};
      get_symbol(s, lib, "cuFileReadAsync");
      stream_available = true;
    } catch (const std::runtime_error&) {
    }
#endif

    // cuFile is supposed to open and close the driver automatically but because of a bug in
    // CUDA 11.8, it sometimes segfault. See <https://github.com/rapidsai/kvikio/issues/159>.
    CUfileError_t const error = DriverOpen();
    if (error.err != CU_FILE_SUCCESS) {
      throw std::runtime_error(std::string{"cuFile error at: "} + __FILE__ + ":" +
                               KVIKIO_STRINGIFY(__LINE__) + ": " +
                               cufileop_status_error(error.err));
    }
  }
  ~cuFileAPI()
  {
    CUfileError_t const error = DriverClose();
    if (error.err != CU_FILE_SUCCESS) {
      std::cerr << "Unable to close GDS file driver: " << cufileop_status_error(error.err)
                << std::endl;
    }
  }

 public:
  cuFileAPI(cuFileAPI const&)      = delete;
  void operator=(cuFileAPI const&) = delete;

  static cuFileAPI& instance()
  {
    static cuFileAPI _instance;
    return _instance;
  }
};

#endif

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
 * @brief Check if cuFile's batch and stream API is available
 *
 * Technically, the batch API is available in CUDA 12.1 but since there is no good
 * way to check CUDA version using the driver API, we check for the existing of the
 * `cuFileReadAsync` symbol, which is defined in CUDA 12.2+.
 *
 * @return The boolean answer
 */
#if defined(KVIKIO_CUFILE_STREAM_API_FOUND) && defined(KVIKIO_CUFILE_STREAM_API_FOUND)
inline bool is_batch_and_stream_available()
{
  try {
    return is_cufile_available() && cuFileAPI::instance().stream_available;
  } catch (const std::runtime_error&) {
    return false;
  }
}
#else
constexpr bool is_batch_and_stream_available() { return false; }
#endif

}  // namespace kvikio
