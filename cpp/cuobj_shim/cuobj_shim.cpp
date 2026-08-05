/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// extern "C" shim over NVIDIA cuObject (libcuobjclient), implementing the
// "manual RDMA token" pattern (cuObjClient API spec section 1.12.4). cuObject
// is a C++ library whose client object requires an I/O-ops callback table at
// construction, so it cannot be loaded directly through KvikIO's dlopen shim
// (which resolves plain C symbols, like cuFileAPI does for libcufile). This
// thunk exposes a small C ABI that KvikIO's cuObjAPI dlopens at runtime; only
// this translation unit is compiled against cuobjclient.h and links
// libcuobjclient, so the main KvikIO library keeps cuObject as an optional
// runtime dependency.
//
// Built into libkvikio_cuobj_shim.so only when cuobjclient.h is found (see
// cpp/CMakeLists.txt). Point KVIKIO_CUOBJ_SHIM at the result, or install it on
// the loader path.

#include <cuobjclient.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace {

// cuObject get/put callbacks are unused in the manual-token pattern (KvikIO
// drives the S3 request out of band over libcurl), but the cuObjClient
// constructor requires an ops table. Provide stubs.
ssize_t cuobj_stub_get(const void* /*handle*/,
                       char* /*ptr*/,
                       size_t /*size*/,
                       loff_t /*offset*/,
                       const cufileRDMAInfo_t*)
{
  return -EOPNOTSUPP;
}

ssize_t cuobj_stub_put(const void* /*handle*/,
                       const char* /*ptr*/,
                       size_t /*size*/,
                       loff_t /*offset*/,
                       const cufileRDMAInfo_t*)
{
  return -EOPNOTSUPP;
}

cuObjClient* get_client()
{
  static CUObjOps_t ops = {cuobj_stub_get, cuobj_stub_put};
  static cuObjClient client(ops);
  return &client;
}

// Descriptors returned by cuMemObjGetRDMAToken are owned by cuObject and must
// be released via cuMemObjPutRDMAToken. Keep the original pointer keyed by
// descriptor value so the caller can free it by the C string it received,
// without owning the lifetime.
std::mutex& token_mutex()
{
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, char*>& token_registry()
{
  static std::unordered_map<std::string, char*> r;
  return r;
}

}  // namespace

extern "C" {

int kvikio_cuobj_available() { return get_client()->isConnected() ? 1 : 0; }

int kvikio_cuobj_register_buffer(void* ptr, size_t size)
{
  return get_client()->cuMemObjGetDescriptor(ptr, size) == CU_OBJ_SUCCESS ? 0 : -1;
}

int kvikio_cuobj_deregister_buffer(void* ptr)
{
  return get_client()->cuMemObjPutDescriptor(ptr) == CU_OBJ_SUCCESS ? 0 : -1;
}

// Returns the descriptor string for [offset, offset+size) of the registered
// buffer, or nullptr on failure. is_put selects PUT (1) vs GET (0). The caller
// must release it via kvikio_cuobj_put_rdma_token after the request completes.
const char* kvikio_cuobj_get_rdma_token(void* ptr, size_t size, size_t offset, int is_put)
{
  char* desc            = nullptr;
  cuObjOpType_t op      = is_put ? CUOBJ_PUT : CUOBJ_GET;
  cuObjErr_t const stat = get_client()->cuMemObjGetRDMAToken(ptr, size, offset, op, &desc);
  if (stat != CU_OBJ_SUCCESS || desc == nullptr) { return nullptr; }
  std::lock_guard<std::mutex> lock(token_mutex());
  token_registry()[std::string(desc)] = desc;
  return desc;
}

int kvikio_cuobj_put_rdma_token(const char* token)
{
  if (token == nullptr) { return -1; }
  char* desc = nullptr;
  {
    std::lock_guard<std::mutex> lock(token_mutex());
    auto it = token_registry().find(std::string(token));
    if (it == token_registry().end()) { return 0; }
    desc = it->second;
    token_registry().erase(it);
  }
  return get_client()->cuMemObjPutRDMAToken(desc) == CU_OBJ_SUCCESS ? 0 : -1;
}

}  // extern "C"
