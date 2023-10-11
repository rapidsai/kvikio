# KvikIO: High Performance File IO

## Summary

KvikIO is a Python and C++ library for high performance file IO. It provides C++ and Python
bindings to [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html),
which enables [GPUDirect Storage (GDS)](https://developer.nvidia.com/blog/gpudirect-storage/).
KvikIO also works efficiently when GDS isn't available and can read/write both host and device data seamlessly.
The C++ library is header-only making it easy to include in [existing projects](https://github.com/rapidsai/kvikio/blob/HEAD/cpp/examples/downstream/).


### Features

* Object oriented API of [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) with C++/Python exception handling.
* A Python [Zarr](https://zarr.readthedocs.io/en/stable/) backend for reading and writing GPU data to file seamlessly.
* Concurrent reads and writes using an internal thread pool.
* Non-blocking API.
* Handle both host and device IO seamlessly.
* Provides Python bindings to [nvCOMP](https://github.com/NVIDIA/nvcomp).

### Documentation
 * Python: <https://docs.rapids.ai/api/kvikio/nightly/>
 * C++: <https://docs.rapids.ai/api/libkvikio/nightly/>
