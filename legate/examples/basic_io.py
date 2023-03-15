# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cunumeric as num
import legate_kvikio as kvikio

from legate.core import get_legate_runtime


def main(path):
    a = num.arange(1_000_000)
    f = kvikio.CuFile(path, "w")
    # Write whole array to file
    f.write(a)
    f.close()

    # In order to make sure the file has been written before the following
    # reads execute, we insert a fence between the write and reads.
    # Notice, this call isn't blocking.
    get_legate_runtime().issue_execution_fence(block=False)

    b = num.empty_like(a)
    f = kvikio.CuFile(path, "r")
    # Read whole array from file
    f.read(b)
    assert all(a == b)

    # Use contexmanager
    c = num.empty_like(a)
    with kvikio.CuFile(path, "r") as f:
        f.read(c)
    assert all(a == c)
    assert a.sum() == b.sum()
    assert b.sum() == c.sum()
    print("sum: ", c.sum())


if __name__ == "__main__":
    main("/tmp/legate-kvikio-hello-world-file")
