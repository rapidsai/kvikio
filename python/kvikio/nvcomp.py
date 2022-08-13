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


class nvCompManager:
    """Base class for nvComp Compression Managers.

    Compression managers compress uncompressed data and decompress the result.

    Child types of nvCompManager implement only their constructor, as they each
    take different options to build. The rest of their implementation is
    in nvCompManager.

    nvCompManager also keeps all of the options for its child types.
    """

    _manager: _lib._nvcompManager = None
    config: dict = {}
    decompression_config: dict = {}

    # This is a python option: What type was the data when it was passed in?
    # This is used only for returning a decompressed view of the original
    # datatype. Untested so far.
    input_type = cp.int8

    # Default options exist for every option type for every class that inherits
    # from nvCompManager, which takes advantage of the below property-setting
    # code.
    stream: cp.cuda.Stream = cp.cuda.Stream()
    chunk_size: int = 1 << 16
    data_type: _lib.pyNvcompType_t = _lib.pyNvcompType_t.pyNVCOMP_TYPE_CHAR
    # Some classes have this defined as type, some as data_type.
    type: _lib.pyNvcompType_t = _lib.pyNvcompType_t.pyNVCOMP_TYPE_CHAR
    device_id: int = 0

    def __init__(self, kwargs):
        """Stores the results of all input arguments as class members.

        This code does type correction, fixing inputs to have an expected
        shape before calling one of the nvCompManager methods on a child
        class.

        Special case: Convert data_type to a _lib.pyNvcompType_t
        """
        if kwargs.get("data_type"):
            if not isinstance(kwargs["data_type"], _lib.pyNvcompType_t):
                kwargs["input_type"] = kwargs.get("data_type")
                kwargs["data_type"] = cp_to_nvcomp_dtype(
                    cp.dtype(kwargs["data_type"]).type
                )
        # Special case: Convert type to a _lib.pyNvcompType_t
        if kwargs.get("type"):
            if not isinstance(kwargs["type"], _lib.pyNvcompType_t):
                kwargs["input_type"] = kwargs.get("type")
                kwargs["type"] = cp_to_nvcomp_dtype(cp.dtype(kwargs["type"]).type)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def compress(self, data: cp.ndarray) -> cp.ndarray:
        """Compress a buffer.

        Parameters
        ----------
        data: cp.ndarray
            A GPU buffer of data to compress.

        Returns
        -------
        cp.ndarray
            A GPU buffer of compressed bytes.
        """
        # TODO: An option: check if incoming data size matches the size of the
        # last incoming data, and reuse temp and out buffer if so.
        data_size = data.size * data.itemsize
        self.config = self._manager.configure_compression(data_size)
        self.compress_out_buffer = cp.zeros(
            self.config["max_compressed_buffer_size"], dtype="uint8"
        )
        size = self._manager.compress(data, self.compress_out_buffer)
        return self.compress_out_buffer[0:size]

    def decompress(self, data: cp.ndarray) -> cp.ndarray:
        """Decompress a GPU buffer.

        Parameters
        ----------
        data: cp.ndarray
            A GPU buffer of data to decompress.

        Returns
        -------
        cp.ndarray
            An array of `self.dtype` produced after decompressing the input argument.
        """
        self.decompression_config = (
            self._manager.configure_decompression_with_compressed_buffer(data)
        )
        decomp_buffer = cp.zeros(
            self.decompression_config["decomp_data_size"], dtype="uint8"
        )
        self._manager.decompress(decomp_buffer, data)
        return decomp_buffer.view(self.input_type)

    def configure_compression(self, data_size: int) -> dict:
        """Return the compression configuration object.

        Parameters
        ----------
        data_size: int
            The size of the buffer that is staged to be compressed.

        Returns
        -------
        dict {
            "uncompressed_buffer_size": The size of the input data
            "max_compressed_buffer_size": The maximum size of the compressed data. The
                size of the buffer that must be allocated before calling compress.
            "num_chunks": The number of configured chunks to compress the data over
        }
        """
        return self._manager.configure_compression(data_size)

    def configure_decompression_with_compressed_buffer(
        self, data: cp.ndarray
    ) -> cp.ndarray:
        """Return the decompression configuration object.

        Parameters
        ----------
        data: cp.ndarray
            A GPU buffer of previously compressed data.

        Returns
        -------
        dict {
            "decomp_data_size": The size of each decompression chunk.
            "num_chunks": The number of chunks that the decompressed data is returned
            in.
        }
        """
        return self._manager.configure_decompression_with_compressed_buffer(data)

    def get_required_scratch_buffer_size(self) -> int:
        """Return the size of the optional scratch buffer.

        Returns
        -------
        int
        """
        return self._manager.get_required_scratch_buffer_size()

    def set_scratch_buffer(self, new_scratch_buffer: cp.ndarray) -> None:
        """Use a pre-allocated buffer for compression.

        Use a GPU-allocated buffer that will be used for compression
        temporary storage instead of allowing the library to create the
        scratch buffer.
        Can reduce memory usage.

        Parameters
        ----------
        new_scratch_buffer
        The buffer that you allocated on the GPU for compressor temporary
        storage.

        Returns
        -------
        cp.ndarray
        """
        return self._manager.set_scratch_buffer(new_scratch_buffer)

    def get_compressed_output_size(self, comp_buffer: cp.ndarray) -> int:
        """Return the actual size of compression result.

        Returns the number of bytes that should be copied out of
        `comp_buffer`.

        Parameters
        ----------
        comp_buffer: cp.ndarray
            A GPU buffer that has been previously compressed.

        Returns
        -------
        int
        """
        return self._manager.get_compressed_output_size(comp_buffer)


class ANSManager(nvCompManager):
    def __init__(self, **kwargs):
        raise NotImplementedError("ANSManager is not supported yet in python.")


class BitcompManager(nvCompManager):
    def __init__(self, **kwargs):
        raise NotImplementedError("BitcompManager is not supported yet in python.")


class CascadedManager(nvCompManager):
    def __init__(self, **kwargs):
        """Initialize a CascadedManager for a specific dtype.

        Parameters
        ----------
        dtype: cp.dtype
            The dtype of the input buffer to be compressed.
        num_RLEs: int
            Number of Run-Length Encoders to use, see [algorithms overview.md](
                https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md#run-length-encoding-rle)  # noqa: E501
        num_deltas: int
            Number of Delta Encoders to use, see [algorithms overview.md](
                https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md#delta-encoding)  # noqa: E501
        use_bp: bool
            Enable Bitpacking, see [algorithms overview.md](
                https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md#bitpacking)  # noqa: E501
        """
        super().__init__(kwargs)
        default_options = {
            "chunk_size": 1 << 12,
            "type": np.int32,
            "num_RLEs": 2,
            "num_deltas": 1,
            "use_bp": True,
        }
        # Replace any options that may have been excluded, they are not optional.
        for k, v in default_options.items():
            try:
                getattr(self, k)
            except Exception:
                setattr(self, k, v)

        self.options = {
            "chunk_size": self.chunk_size,
            "type": self.type,
            "num_RLEs": self.num_RLEs,
            "num_deltas": self.num_deltas,
            "use_bp": self.use_bp,
        }
        self._manager = _lib._CascadedManager(
            default_options, self.stream, self.device_id
        )


class GDeflateManager(nvCompManager):
    def __init__(self, **kwargs):
        raise NotImplementedError("GDeflateManager is not supported yet in python.")


class LZ4Manager(nvCompManager):
    def __init__(self, **kwargs):
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
        super().__init__(kwargs)
        self._manager = _lib._LZ4Manager(
            self.chunk_size, self.data_type.value, self.stream, self.device_id
        )


class SnappyManager(nvCompManager):
    def __init__(self, **kwargs):
        """Create a GPU SnappyCompressor object.

        Used to compress and decompress GPU buffers.

        Parameters
        ----------
        chunk_size: int (optional)
        stream: cudaStream_t (optional)
            Which CUDA stream to perform the operation on
        device_id: int (optional)
            Specify which device_id on the node to use
        """
        super().__init__(kwargs)
        self._manager = _lib._SnappyManager(
            self.chunk_size, self.stream, self.device_id
        )


class ManagedDecompressionManager(nvCompManager):
    def __init__(self, compressed_buffer):
        """Create a Managed compressor using the
        create_manager factory method.

        This function is used in order to automatically
        identify which compression algorithm was used on
        an input buffer.

        It returns a ManagedDecompressionManager that can
        then be used normally to decompress the unknown
        compressed binary data, or compress other data
        into the same format.

        Parameters
        ----------
        compressed_buffer: cp.ndarray
            A buffer of compressed bytes of unknown origin.
        """
        super().__init__({})
        self._manager = _lib._ManagedManager(compressed_buffer)
