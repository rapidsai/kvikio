# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cunumeric as num
import legate_kvikio as kvikio

from legate.core import get_legate_runtime


def main(path):
    a = num.arange(1_000_000)
    f = kvikio.CuFile(path, "w+")
    f.write(a)

    # In order to make sure the file has been written before the following
    # read execute, we insert a fence between the write and read.
    # Notice, this call isn't blocking.
    get_legate_runtime().issue_execution_fence(block=False)

    b = num.empty_like(a)
    f.read(b)
    f.close()

    # In order to make sure the file has been written before re-opening
    # it for reading, we block the execution.
    get_legate_runtime().issue_execution_fence(block=True)

    c = num.empty_like(a)
    with kvikio.CuFile(path, "r") as f:
        f.read(c)

    # They should all be identical
    assert all(a == b)
    assert all(a == c)
    print("sum: ", c.sum())


if __name__ == "__main__":
    main("/tmp/legate-kvikio-hello-world-file")
