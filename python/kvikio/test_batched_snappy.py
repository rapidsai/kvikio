#!/home/tcomer/mnt/NVIDIA/rapids-docker/compose/etc/conda/cuda_11.5/envs/rapids/bin/python

import cudf
import cupy as cp
import numpy as np

def test():
    num_arrays = 2
    size = 10
    snappy = cudf.core.tools.nvcomp.SnappyCompressor()
    data = cp.array(list(range(0,320)), dtype=np.uint8)
    print(snappy.compress(data)) 

if __name__ == '__main__':
    test()
