/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
  cuFileAPI();

#ifdef KVIKIO_CUFILE_FOUND
  // Notice, we have to close the driver at program exit (if we opened it) even though we are
  // not allowed to call CUDA after main[1]. This is because, cuFile will segfault if the
  // driver isn't closed on program exit i.e. we are doomed if we do, doomed if we don't, but
  // this seems to be the lesser of two evils.
  // [1] <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
  ~cuFileAPI();
#endif

 public:
  cuFileAPI(cuFileAPI const&)       = delete;
  void operator=(cuFileAPI const&)  = delete;
  cuFileAPI(cuFileAPI const&&)      = delete;
  void operator=(cuFileAPI const&&) = delete;

  KVIKIO_EXPORT static cuFileAPI& instance();

  /**
   * @brief Open the cuFile driver
   *
   * cuFile allows multiple calls to `cufileDriverOpen()`, only the first call opens
   * the driver, but every call should have a matching call to `cufileDriverClose()`.
   */
  void driver_open();

  /**
   * @brief Close the cuFile driver
   */
  void driver_close();
};

/**
 * @brief Check if the cuFile library is available
 *
 * Notice, this doesn't check if the runtime environment supports cuFile.
 *
 * @return The boolean answer
 */
#ifdef KVIKIO_CUFILE_FOUND
bool is_cufile_library_available() noexcept;
#else
constexpr bool is_cufile_library_available() noexcept { return false; }
#endif

/**
 * @brief Check if the cuFile is available and expected to work
 *
 * Besides checking if the cuFile library is available, this also checks the
 * runtime environment.
 *
 * @return The boolean answer
 */
bool is_cufile_available() noexcept;

/**
 * @brief Get cufile version (or zero if older than v1.8).
 *
 * The version is returned as (1000*major + 10*minor). E.g., cufile v1.8.0 would
 * be represented by 1080.
 *
 * Notice, this is not the version of the CUDA toolkit. cufile is part of the
 * toolkit but follows its own version scheme.
 *
 * @return The version (1000*major + 10*minor) or zero if older than 1080.
 */
#ifdef KVIKIO_CUFILE_FOUND
int cufile_version() noexcept;
#else
constexpr int cufile_version() noexcept { return 0; }
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
bool is_batch_api_available() noexcept;

/**
 * @brief Check if cuFile's stream (async) API is available.
 *
 * Since `cuFileGetVersion()` first became available in cufile v1.8 (CTK v12.3),
 * this function returns false for versions older than v1.8 even though the stream
 * API became available in v1.7.
 *
 * @return The boolean answer
 */
bool is_stream_api_available() noexcept;

}  // namespace kvikio
