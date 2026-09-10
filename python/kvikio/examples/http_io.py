# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import tempfile

import cupy
import numpy

import kvikio
from kvikio.utils import LocalHttpServer


def main(tmpdir: pathlib.Path):
    monitor = kvikio.SummaryMonitor()

    a = cupy.arange(100)
    a.tofile(tmpdir / "myfile")
    b = cupy.empty_like(a)

    # Start a local server that serves files in `tmpdir`
    with LocalHttpServer(root_path=tmpdir) as server:
        # Open remote file from a http url
        with kvikio.RemoteFile.open_http(f"{server.url}/myfile") as f:
            # KvikIO fetch the file size
            assert f.nbytes() == a.nbytes
            # Read the remote file into `b` as if it was a local file.
            f.read(b)
            assert all(a == b)
            # We can also read into host memory seamlessly
            a = cupy.asnumpy(a)
            c = numpy.empty_like(a)
            f.read(c)
            assert all(a == c)

    summary = monitor.get()
    print(summary)
    print(
        f"{summary.busy_bytes_per_sec / 1e6:.2f} MB/s while fetching, "
        f"busy {summary.busy_fraction * 100:.1f} % of the time"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        main(pathlib.Path(tmpdir))
