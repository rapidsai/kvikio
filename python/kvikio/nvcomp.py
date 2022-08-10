# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from enum import Enum

import cupy as cp
import numpy as np

import kvikio._lib.libnvcomp as _lib

_dtype_map = {
    cp.dtype("int8"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_CHAR,
    cp.dtype("uint8"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_UCHAR,
    cp.dtype("int16"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_SHORT,
    cp.dtype("uint16"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_USHORT,
    cp.dtype("int32"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_INT,
    cp.dtype("uint32"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_UINT,
    cp.dtype("int64"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_LONGLONG,
    cp.dtype("uint64"): _lib.pyNvcompType_t.pyNVCOMP_TYPE_ULONGLONG,
}


def cp_to_nvcomp_dtype(in_type: cp.dtype) -> Enum:
    """Convert np/cp dtypes to nvcomp integral dtypes.

    Parameters
    ----------
    in_type
        A type argument that can be used to initialize a cupy/numpy dtype.

    Returns
    -------
    int
        The value of the NVCOMP_TYPE for supported dtype.
    """
    cp_type = cp.dtype(in_type)
    return _dtype_map[cp_type]


class CascadedCompressor:
    def __init__(
        self,
        dtype: cp.dtype,
        num_RLEs: int = 1,
        num_deltas: int = 1,
        use_bp: bool = True,
    ):
        """Initialize a CascadedCompressor and Decompressor for a specific dtype.

        Parameters
        ----------
        dtype: cp.dtype
            The dtype of the input buffer to be compressed.
        num_RLEs: int
            Number of Run-Length Encoders to use, see [algorithms overview.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md#run-length-encoding-rle)  # noqa: E501
        num_deltas: int
            Number of Delta Encoders to use, see [algorithms overview.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md#delta-encoding)  # noqa: E501
        use_bp: bool
            Enable Bitpacking, see [algorithms overview.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md#bitpacking)  # noqa: E501
        """
        self.dtype = dtype
        self.compressor = _lib._CascadedCompressor(
            cp_to_nvcomp_dtype(self.dtype).value,
            num_RLEs,
            num_deltas,
            use_bp,
        )
        self.decompressor = _lib._CascadedDecompressor()
        self.s = cp.cuda.Stream()

    def compress(self, data: cp.ndarray) -> cp.ndarray:
        """Compress a buffer.

        Returns
        -------
        cp.ndarray
            A GPU buffer of compressed bytes.
        """
        # TODO: An option: check if incoming data size matches the size of the
        # last incoming data, and reuse temp and out buffer if so.
        data_size = data.size * data.itemsize
        self.compress_temp_size = np.zeros((1,), dtype=np.int64)
        self.compress_out_size = np.zeros((1,), dtype=np.int64)
        self.compressor.configure(
            data_size, self.compress_temp_size, self.compress_out_size
        )
        self.compress_temp_buffer = cp.zeros(
            self.compress_temp_size, dtype=np.uint8
        )
        self.compress_out_buffer = cp.zeros(
            self.compress_out_size, dtype=np.uint8
        )
        self.compressor.compress_async(
            data,
            data_size,
            self.compress_temp_buffer,
            self.compress_temp_size,
            self.compress_out_buffer,
            self.compress_out_size,
            self.s.ptr,
        )
        return self.compress_out_buffer[: self.compress_out_size[0]]

    def decompress(self, data: cp.ndarray) -> cp.ndarray:
        """Decompress a GPU buffer.

        Returns
        -------
        cp.ndarray
            An array of `self.dtype` produced after decompressing the input argument.
        """
        # TODO: logic to reuse temp buffer if it is large enough
        data_size = data.size * data.itemsize
        self.decompress_temp_size = np.zeros((1,), dtype=np.int64)
        self.decompress_out_size = np.zeros((1,), dtype=np.int64)

        self.decompressor.configure(
            data,
            data_size,
            self.decompress_temp_size,
            self.decompress_out_size,
            self.s.ptr,
        )

        self.decompress_temp_buffer = cp.zeros(
            self.decompress_temp_size, dtype=np.uint8
        )
        self.decompress_out_buffer = cp.zeros(
            self.decompress_out_size, dtype=np.uint8
        )
        self.decompressor.decompress_async(
            data,
            data_size,
            self.decompress_temp_buffer,
            self.decompress_temp_size,
            self.decompress_out_buffer,
            self.decompress_out_size,
            self.s.ptr,
        )
        return self.decompress_out_buffer.view(self.dtype)


class nvCompManager:
    def __init__(self, kwargs):
        if kwargs.get("data_type"):
            if isinstance(kwargs["data_type"], str):
                kwargs["data_type"] = cp_to_nvcomp_dtype(
                    cp.dtype(kwargs["data_type"]).type
                )
        for k, v in kwargs.items():
            setattr(self, k, v)


class LZ4Compressor(nvCompManager):
    def __init__(
        self,
        chunk_size: int = 1 << 16,
        data_type: _lib.pyNvcompType_t = _lib.pyNvcompType_t.pyNVCOMP_TYPE_CHAR,
        stream: cp.cuda.Stream = cp.cuda.Stream(),
        device_id: int = 0,
    ):
        """Create a GPU LZ4Compressor object.

        Used to compress and decompress GPU buffers of a specific dtype.

        Parameters
        ----------
        chunk_size: int
        data_type: pyNVCOMP_TYPE
        stream: cudaStream_t (optional)
            Which CUDA stream to perform the operation on
        device_id: int (optional)
            Specify which device_id on the node to use
        """
        super().__init__(
            {
                "chunk_size": chunk_size,
                "data_type": data_type,
                "stream": stream,
                "device_id": device_id,
            }
        )
        self.compressor = _lib._LZ4Compressor(
            self.chunk_size, self.data_type.value, self.stream, self.device_id
        )

    def compress(self, data: cp.ndarray) -> cp.ndarray:
        """Compress a buffer.

        Returns
        -------
        cp.ndarray
            A GPU buffer of compressed bytes.
        """
        # TODO: An option: check if incoming data size matches the size of the
        # last incoming data, and reuse temp and out buffer if so.
        print("def compress(...)")
        data_size = data.size * data.itemsize
        self.config = self.compressor.configure_compression(data_size)
        self.compress_out_buffer = cp.zeros(
            self.config["max_compressed_buffer_size"], dtype="uint8"
        )
        size = self.compressor.compress(data, self.compress_out_buffer)
        return self.compress_out_buffer[0:size]

    def decompress(self, data: cp.ndarray) -> cp.ndarray:
        """Decompress a GPU buffer.

        Returns
        -------
        cp.ndarray
            An array of `self.dtype` produced after decompressing the input argument.
        """
        # TODO: logic to reuse temp buffer if it is large enough
        data_size = data.size * data.itemsize
        self.decompress_temp_size = np.zeros((1,), dtype=np.int64)
        self.decompress_out_size = np.zeros((1,), dtype=np.int64)

        self.decompressor.configure(
            data,
            data_size,
            self.decompress_temp_size,
            self.decompress_out_size,
            self.s.ptr,
        )

        self.decompress_temp_buffer = cp.zeros(
            self.decompress_temp_size, dtype=np.uint8
        )
        self.decompress_out_buffer = cp.zeros(
            self.decompress_out_size, dtype=np.uint8
        )
        self.decompressor.decompress_async(
            data,
            data_size,
            self.decompress_temp_buffer,
            self.decompress_temp_size,
            self.decompress_out_buffer,
            self.decompress_out_size,
            self.s.ptr,
        )
        return self.decompress_out_buffer.view(self.dtype)
