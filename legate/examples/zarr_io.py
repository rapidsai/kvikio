# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cunumeric as num

import legate.core
import legate_kvikio.zarr


def zarr_io(dirname):
    a = num.arange(10000).reshape(100, 100)

    # Write array to a Zarr file by chunks of 10x10.
    legate_kvikio.zarr.write_array(a, dirname, chunks=(10, 10))

    # Block until done writing.
    legate.core.get_legate_runtime().issue_execution_fence(block=True)

    # Read array from a Zarr file.
    b = legate_kvikio.zarr.read_array(dirname)

    # They should be equal
    assert (a == b).all()


if __name__ == "__main__":
    zarr_io("/tmp/legate-kvikio-zarr-io")
