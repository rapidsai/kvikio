
import cupy as cp
import numpy as np
import kvikio._lib.nvcomp as _lib


def make_ptr_collection(data):
    return [_lib.__get_ptr(x) for x in data]


def get_ptr(x):
    return _lib.__get_ptr(x)


class CascadedCompressor:
    def __init__(self):
        self.compressor = _lib._CascadedCompressor()
        self.decompressor = _lib._CascadedDecompressor()
        self.s = cp.cuda.Stream()

    def compress(self, data):
        # TODO: An option: check if incoming data size matches the size of the
        # last incoming data, and reuse temp and out buffer if so.
        data_size = data.size * data.itemsize
        self.compress_temp_size = cp.zeros(1, dtype=np.uint64)
        self.compress_out_size = cp.zeros(1, dtype=np.uint64)
        self.compressor.configure(
            data_size,
            self.compress_temp_size,
            self.compress_out_size
        )

        self.compress_temp_buffer = cp.zeros(self.temp_size, dtype=np.uint8)
        self.compress_out_buffer = cp.zeros(self.out_size, dtype=np.uint8)
        self.compressor.compress_async(
            data,
            data_size,
            self.compress_temp_buffer,
            self.compress_temp_bytes,
            self.compress_out_buffer,
            self.compress_out_size,
            self.s.ptr
        )
        return self.compress_out_buffer, self.compress_out_size

    def decompress(self, data):
        # TODO: logic to reuse temp buffer if it is large enough
        data_size = data.size * data.itemsize
        self.decompress_temp_size = cp.zeros(1, dtype=np.uint64)
        self.decompress_out_size = cp.zeros(1, dtype=np.uint64)

        self.decompressor.configure(
            data,
            data_size,
            self.decompress_temp_size,
            self.decompress_out_size,
            self.s.ptr
        )

        self.decompress_temp_buffer = cp.zeros(self.temp_size, dtype=np.uint8)
        self.decompress_out_buffer = cp.zeros(self.out_size, dtype=np.uint8)
        self.decompress_async(
            data,
            data_size,
            self.decompress_temp_buffer,
            self.decompress_temp_size,
            self.decompress_out_buffer,
            self.decompress_out_size,
            self.s.ptr
        )


class SnappyCompressor:
    def __init__(self):
        self.compressor = _lib.LibSnappyCompressor()

    def compress(self, data):
        # convert data to pointers
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

        # allocate output buffers
        output_buffers = cp.zeros(512, dtype=cp.int8)
        output_buffers_ptr = get_ptr(output_buffers)
        # allocate output sizes
        output_size = cp.zeros(1, dtype=cp.uint64)
        # call compress with sizes and buffers

        s = cp.cuda.Stream()

        data = cp.array(list(range(0, 256)), dtype=np.uint8)
        data_ptr = cp.array([get_ptr(data)], dtype=np.uint64)
        data_sizes = cp.array([256], dtype=np.uint64)
        num_chunks = cp.array([1], dtype=np.uint64)
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
