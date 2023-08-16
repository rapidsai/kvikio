# KvikIO: C++ and Python bindings to cuFile

## Summary

This provides C++ and Python bindings to cuFile, which enables GPUDirect Storage (GDS).
KvikIO also works efficiently when GDS isn't available and can read/write both host and
device data seamlessly.

### Features

* Object Oriented API.
* Exception handling.
* Concurrent reads and writes using an internal thread pool.
* Non-blocking API.
* Python Zarr reader.
* Handle both host and device IO seamlessly.
* Provides Python bindings to [nvCOMP](https://github.com/NVIDIA/nvcomp).

## Requirements

To install users should have a working Linux machine with CUDA Toolkit
installed (v11.4+) and a working compiler toolchain (C++17 and cmake).

### C++

The C++ bindings are header-only and depends on the CUDA Driver API.
In order to build and run the example code, CMake and the CUDA Runtime
API is required.

### Python

The Python package depends on the following packages:

* cython
* pip
* setuptools
* scikit-build

For nvCOMP, benchmarks, examples, and tests:

* pytest
* numpy
* cupy

## Install

### Conda

Install the stable release from the `rapidsai` channel like:

```
conda create -n kvikio_env -c rapidsai -c conda-forge kvikio
```

Install the `kvikio` conda package from the `rapidsai-nightly` channel like:

```
conda create -n kvikio_env -c rapidsai-nightly -c conda-forge python=3.10 cuda-version=11.8 kvikio
```

If the nightly install doesn't work, set `channel_priority: flexible` in your `.condarc`.

In order to setup a development environment run:
```
conda env create --name kvikio-dev --file conda/environments/all_cuda-118_arch-x86_64.yaml
```

### C++ (build from source)

To build the C++ example run:

```
./build.sh libkvikio
```

Then run the example:

```
./examples/basic_io
```

### Python (build from source)

To build and install the extension run:

```
./build.sh kvikio
```

One might have to define `CUDA_HOME` to the path to the CUDA installation.

In order to test the installation, run the following:

```
pytest tests/
```

And to test performance, run the following:

```
python benchmarks/single-node-io.py
```

## Examples


### Notebooks
 - [How to read and write GPU memory directly to/from Zarr files](notebooks/zarr.ipynb)


### C++

```c++
#include <cstddef>
#include <cuda_runtime.h>
#include <kvikio/file_handle.hpp>
using namespace std;

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
  size_t written = fw.write(a, size);
  fw.close();

  // Read file into `b`
  kvikio::FileHandle fr("test-file", "r");
  size_t read = fr.read(b, size);
  fr.close();

  // Read file into `b` in parallel using 16 threads
  kvikio::default_thread_pool::reset(16);
  {
    kvikio::FileHandle f("test-file", "r");
    future<size_t> future = f.pread(b_dev, sizeof(a), 0);  // Non-blocking
    size_t read = future.get(); // Blocking
    // Notice, `f` closes automatically on destruction.
  }
}
```

### Python

```python
import cupy
import kvikio

a = cupy.arange(100)
f = kvikio.CuFile("test-file", "w")
# Write whole array to file
f.write(a)
f.close()

b = cupy.empty_like(a)
f = kvikio.CuFile("test-file", "r")
# Read whole array from file
f.read(b)
assert all(a == b)

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
    future1.get()  # Wait for first read
    future2.get()  # Wait for second read
assert all(a == d)
```
