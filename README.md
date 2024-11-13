# KvikIO: High Performance File IO

## Summary

KvikIO (pronounced "kuh-VICK-eye-oh", see [here](https://ordnet.dk/ddo_en/dict?query=kvik) for pronunciation of kvik) is a Python and C++ library for high performance file IO. It provides C++ and Python
bindings to [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html),
which enables [GPUDirect Storage (GDS)](https://developer.nvidia.com/blog/gpudirect-storage/).
KvikIO also works efficiently when GDS isn't available and can read/write both host and device data seamlessly.


### Features

* Object oriented API of [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) with C++/Python exception handling.
* A Python [Zarr](https://zarr.readthedocs.io/en/stable/) backend for reading and writing GPU data to file seamlessly.
* Concurrent reads and writes using an internal thread pool.
* Non-blocking API.
* Transparently handles reads and writes to/from memory on both host and device.
* Provides Python bindings to [nvCOMP](https://github.com/NVIDIA/nvcomp).


### Documentation
 * Python: <https://docs.rapids.ai/api/kvikio/nightly/>
 * C++: <https://docs.rapids.ai/api/libkvikio/nightly/>


### Examples

#### Python
```python
import cupy
import kvikio

def main(path):
    a = cupy.arange(100)
    f = kvikio.CuFile(path, "w")
    # Write whole array to file
    f.write(a)
    f.close()

    b = cupy.empty_like(a)
    f = kvikio.CuFile(path, "r")
    # Read whole array from file
    f.read(b)
    assert all(a == b)
    f.close()

    # Use contexmanager
    c = cupy.empty_like(a)
    with kvikio.CuFile(path, "r") as f:
        f.read(c)
    assert all(a == c)

    # Non-blocking read
    d = cupy.empty_like(a)
    with kvikio.CuFile(path, "r") as f:
        future1 = f.pread(d[:50])
        future2 = f.pread(d[50:], file_offset=d[:50].nbytes)
        # Note: must wait for futures before exiting block
        # at which point the file is closed.
        future1.get()  # Wait for first read
        future2.get()  # Wait for second read
    assert all(a == d)


if __name__ == "__main__":
    main("/tmp/kvikio-hello-world-file")
```

#### C++
```c++
#include <cstddef>
#include <future>
#include <cuda_runtime.h>
#include <kvikio/file_handle.hpp>

int main()
{
  // Create two arrays `a` and `b`
  constexpr std::size_t size = 100;
  void *a = nullptr;
  void *b = nullptr;
  cudaMalloc(&a, size);
  cudaMalloc(&b, size);

  // Write `a` to file
  kvikio::FileHandle fw("test-file", "w");
  std::size_t written = fw.write(a, size);
  fw.close();

  // Read file into `b`
  kvikio::FileHandle fr("test-file", "r");
  std::size_t read = fr.read(b, size);
  fr.close();

  // Read file into `b` in parallel using 16 threads
  kvikio::default_thread_pool::reset(16);
  {
    // FileHandles have RAII semantics
    kvikio::FileHandle f("test-file", "r");
    std::future<std::size_t> future = f.pread(b_dev, sizeof(a), 0);  // Non-blocking
    std::size_t read = future.get(); // Blocking
    // Notice, `f` closes automatically on destruction.
  }
}
```
