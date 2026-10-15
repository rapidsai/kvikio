# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

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
    if not all(a == b):
        raise RuntimeError("Read data does not match written data")

    # Use contexmanager
    c = cupy.empty_like(a)
    with kvikio.CuFile(path, "r") as f:
        f.read(c)
    if not all(a == c):
        raise RuntimeError("Context-manager read does not match written data")

    # Non-blocking read
    d = cupy.empty_like(a)
    with kvikio.CuFile(path, "r") as f:
        future1 = f.pread(d[:50])
        future2 = f.pread(d[50:], file_offset=d[:50].nbytes)
        future1.get()  # Wait for first read
        future2.get()  # Wait for second read
    if not all(a == d):
        raise RuntimeError("Non-blocking read does not match written data")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="kvikio-hello-world-") as tmp_dir:
        main(str(Path(tmp_dir) / "hello-world-file"))
