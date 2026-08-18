# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy

import kvikio


def main(path):
    # Statistics are off until a monitor exists. This one counts every operation below.
    monitor = kvikio.SummaryMonitor()

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

    # Five calls, one write and four reads, however many reads KvikIO issued underneath.
    # the two `pread()`s above are one operation each, not one per thread-pool task.
    summary = monitor.get()
    assert summary.num_ops == 5
    print(summary)
    print(f"{summary.bytes_transferred} bytes in {summary.num_ops} operations")


if __name__ == "__main__":
    main("/tmp/kvikio-hello-world-file")
