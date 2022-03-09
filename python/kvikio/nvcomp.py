
import cupy as cp
import numpy as np
import kvikio._lib as _lib

def make_ptr_collection(data):
    return [__get_ptr(x) for x in data]

def get_ptr(x):
    return __get_ptr(x)

class SnappyCompressor:
    def __init__(self):
        self.compressor = _lib.LibSnappyCompressor()

    def compress(self, data):
        # convert data to pointers
        data_ptrs = get_ptr(data)
        data_sizes = cp.array([256], dtype=cp.uint64)

        num_chunks = 1
        max_chunk_size = 256
        format_opts = 0
        # get output sizke
        max_compressed_chunk_size = np.zeros(1, dtype=np.uint64)
        self.compressor._get_compress_max_output_chunk_size(
                max_chunk_size, max_compressed_chunk_size, format_opts)
        temp_size = np.zeros(1, dtype=np.uint64)
        # get temp size
        result = self.compressor._get_compress_temp_size(
                num_chunks,
                max_compressed_chunk_size,
                temp_size,
                format_opts
                )
        max_compressed_chunk_size = 512

        # allocate temp size
        temp_bytes = cp.zeros(temp_size, dtype=cp.int8)

        # allocate output buffers
        output_buffers = cp.zeros(512, dtype=cp.int8)
        output_buffers_ptr = get_ptr(output_buffers)
        # allocate output sizes
        output_size = cp.zeros(1, dtype=cp.uint64)
        # call compress with sizes and buffers

        s = cp.cuda.Stream()

        max_size = cp.array([512], dtype=cp.uint64)

        data = cp.array(list(range(0,256)), dtype=np.uint8)
        data_ptr = cp.array([get_ptr(data)], dtype=np.uint64)
        data_sizes = cp.array([256], dtype=np.uint64)
        num_chunks = cp.array([1], dtype=np.uint64)#cp.array([1], dtype=np.uint64)
        output_buffers = cp.zeros(1024, dtype=np.uint8)
        output_buffers_ptr = cp.array([get_ptr(output_buffers)], dtype=np.uint64)
        output_sizes = cp.zeros(1, dtype=np.uint64)
        format_opts = 0

        result = self.compressor._compress(
                data_ptr,
                data_sizes,
                np.array([0]),
                num_chunks,
                0,
                0,
                output_buffers_ptr,
                output_sizes,
                format_opts,
                s.ptr
                )
       
        print(output_size[0].get())

        return result
