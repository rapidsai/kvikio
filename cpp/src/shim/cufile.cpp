/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>

#include <kvikio/error.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio {

#ifdef KVIKIO_CUFILE_FOUND
cuFileAPI::cuFileAPI()
{
  void* lib = load_library("libcufile.so.0");
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

  // Some symbols were introduced in later versions, so version guards are required.
  // Note: `version` is 0 for cuFile versions prior to v1.8 because `cuFileGetVersion`
  // did not exist. As a result, the batch and stream APIs are not loaded in versions
  // 1.6 and 1.7, respectively, even though they are available. This trade-off is made
  // for improved robustness.
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
}
#else
cuFileAPI::cuFileAPI() { KVIKIO_FAIL("KvikIO not compiled with cuFile.h", std::runtime_error); }
#endif

cuFileAPI& cuFileAPI::instance()
{
  static cuFileAPI _instance;
  return _instance;
}

void cuFileAPI::driver_open()
{
  CUfileError_t const error = DriverOpen();
  KVIKIO_EXPECT(error.err == CU_FILE_SUCCESS,
                std::string{"Unable to open GDS file driver: "} + cufileop_status_error(error.err),
                std::runtime_error);
}

void cuFileAPI::driver_close()
{
  CUfileError_t const error = DriverClose();
  KVIKIO_EXPECT(error.err == CU_FILE_SUCCESS,
                std::string{"Unable to close GDS file driver: "} + cufileop_status_error(error.err),
                std::runtime_error);
}

#ifdef KVIKIO_CUFILE_FOUND
bool is_cufile_library_available() noexcept
{
  try {
    cuFileAPI::instance();
  } catch (...) {
    return false;
  }
  return true;
}
#endif

bool is_cufile_available() noexcept
{
  return is_cufile_library_available() && run_udev_readable() && !is_running_in_wsl();
}

#ifdef KVIKIO_CUFILE_FOUND
int cufile_version() noexcept
{
  try {
    return cuFileAPI::instance().version;
  } catch (...) {
    return 0;
  }
}
#else
int cufile_version() noexcept { return 0; }
#endif

bool is_batch_api_available() noexcept { return cufile_version() >= 1060; }

bool is_stream_api_available() noexcept { return cufile_version() >= 1070; }

}  // namespace kvikio
