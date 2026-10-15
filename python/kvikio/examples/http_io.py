# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import tempfile

import cupy
import numpy

import kvikio
from kvikio.utils import LocalHttpServer


def main(tmpdir: pathlib.Path):
    a = cupy.arange(100)
    a.tofile(tmpdir / "myfile")
    b = cupy.empty_like(a)

    # Start a local server that serves files in `tmpdir`
    with LocalHttpServer(root_path=tmpdir) as server:
        # Open remote file from a http url
        with kvikio.RemoteFile.open_http(f"{server.url}/myfile") as f:
            # KvikIO fetch the file size
            if f.nbytes() != a.nbytes:
                raise RuntimeError("Remote file size does not match the source array")
            # Read the remote file into `b` as if it was a local file.
            f.read(b)
            if not all(a == b):
                raise RuntimeError("Remote read does not match the source array")
            # We can also read into host memory seamlessly
            a = cupy.asnumpy(a)
            c = numpy.empty_like(a)
            f.read(c)
            if not all(a == c):
                raise RuntimeError("Host-memory read does not match the source array")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        main(pathlib.Path(tmpdir))
