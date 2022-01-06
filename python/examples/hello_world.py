# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cupy

import cufile


def main():
    a = cupy.arange(100)
    f = cufile.CuFile("test-file", "w")
    # Write whole array to file
    f.write(a)
    f.close()

    b = cupy.empty_like(a)
    f = cufile.CuFile("test-file", "r")
    # Read whole array from file
    f.read(b)
    assert all(a == b)


if __name__ == "__main__":
    main()
