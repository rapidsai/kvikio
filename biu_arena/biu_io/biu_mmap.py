# https://docs.python.org/3/library/mmap.html

import mmap
import cupy
import numpy as np


def read(file_name, dev_buf):
    with open(file_name, mode="rb") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            my_data = mmap_obj.read()
            host_buf = np.frombuffer(my_data)
            stream = cupy.cuda.Stream(non_blocking=True)
            cupy.cuda.runtime.memcpyAsync(dev_buf.data.ptr,
                                          host_buf.ctypes.data,
                                          host_buf.nbytes,
                                          cupy.cuda.runtime.memcpyHostToDevice,
                                          stream.ptr)
            cupy.cuda.runtime.streamSynchronize(stream.ptr)
