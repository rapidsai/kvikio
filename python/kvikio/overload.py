# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os.path

import kvikio


class ArrayFromFile:
    def __init__(self, meta_array) -> None:
        self._meta_array = meta_array

    def __array_function__(self, func, types, args, kwargs):
        import numpy

        if func is not numpy.fromfile:
            raise NotImplementedError()

        if len(args):
            filepath = args[0]
            args = args[1:]
        else:
            filepath = kwargs.pop("file")

        # If `filepath` is a File, we get its filepath
        if hasattr(filepath, "fileno"):
            filepath = filepath.name

        if len(args):
            dtype = args[0]
            args = args[1:]
        else:
            dtype = kwargs.pop("dtype", float)

        if args or kwargs:
            raise NotImplementedError("only implements the file and dtype arguments")

        nbytes = os.path.getsize(filepath)
        itemsize = numpy.dtype(dtype).itemsize
        size = nbytes // itemsize
        if nbytes % itemsize != 0:
            raise ValueError(
                f"file size ({nbytes}) not divisible with dtype size ({itemsize})"
            )

        ret = numpy.empty_like(self._meta_array, shape=(size,), dtype=dtype)
        with kvikio.CuFile(filepath, "r") as f:
            f.read(ret)
        return ret
