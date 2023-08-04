# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from abc import ABC, abstractmethod
from enum import IntEnum

from libc.stdint cimport uint32_t, uint64_t, uintptr_t

from kvikio._lib.nvcomp_ll_cxx_api cimport (
    cudaMemcpyKind,
    cudaStream_t,
    nvcompStatus_t,
    nvcompType_t,
)

import cupy
from cupy.cuda.runtime import memcpyAsync


class nvCompStatus(IntEnum):
    Success = nvcompStatus_t.nvcompSuccess,
    ErrorInvalidValue = nvcompStatus_t.nvcompErrorInvalidValue,
    ErrorNotSupported = nvcompStatus_t.nvcompErrorNotSupported,
    ErrorCannotDecompress = nvcompStatus_t.nvcompErrorCannotDecompress,
    ErrorBadChecksum = nvcompStatus_t.nvcompErrorBadChecksum,
    ErrorCannotVerifyChecksums = nvcompStatus_t.nvcompErrorCannotVerifyChecksums,
    ErrorCudaError = nvcompStatus_t.nvcompErrorCudaError,
    ErrorInternal = nvcompStatus_t.nvcompErrorInternal,


class nvCompType(IntEnum):
    CHAR = nvcompType_t.NVCOMP_TYPE_CHAR
    UCHAR = nvcompType_t.NVCOMP_TYPE_UCHAR
    SHORT = nvcompType_t.NVCOMP_TYPE_SHORT
    USHORT = nvcompType_t.NVCOMP_TYPE_USHORT
    INT = nvcompType_t.NVCOMP_TYPE_INT
    UINT = nvcompType_t.NVCOMP_TYPE_UINT
    LONGLONG = nvcompType_t.NVCOMP_TYPE_LONGLONG
    ULONGLONG = nvcompType_t.NVCOMP_TYPE_ULONGLONG
    BITS = nvcompType_t.NVCOMP_TYPE_BITS


class nvCompBatchAlgorithm(ABC):
    """Abstract class that provides interface to nvCOMP batched algorithms."""

    # TODO(akamenev): it might be possible to have a simpler implementation that
    # eilminates the need to have a separate implementation class for each algorithm,
    # potentially using fused types in Cython (similar to C++ templates),
    # but I could not figure out how to do that (e.g. each algorithm API set has
    # a different type for the options and so on).

    def get_compress_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ):
        """Get temporary space required for compression.

        Parameters
        ----------
        batch_size: int
            The number of items in the batch.
        max_uncompressed_chunk_bytes: int
            The maximum size in bytes of a chunk in the batch.

        Returns
        -------
        int
            The size in bytes of the required GPU workspace for compression.
        """
        err, temp_size = self._get_comp_temp_size(
            batch_size,
            max_uncompressed_chunk_bytes
        )
        if err != nvcompStatus_t.nvcompSuccess:
            raise RuntimeError(
                f"Could not get compress temp buffer size, "
                f"error: {nvCompStatus(err)!r}."
            )
        return temp_size

    @abstractmethod
    def _get_comp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ) -> tuple[nvcompStatus_t, size_t]:
        """Algorithm-specific implementation."""
        ...

    def get_compress_chunk_size(self, size_t max_uncompressed_chunk_bytes):
        """Get the maximum size any chunk could compress to in the batch.

        Parameters
        ----------
        max_uncompressed_chunk_bytes: int
            The maximum size in bytes of a chunk in the batch.

        Returns
        -------
        int
            The maximum compressed size in bytes of the largest chunk. That is,
            the minimum amount of output memory required to be given to
            the corresponding *CompressAsync function.
        """
        err, comp_chunk_size = self._get_comp_chunk_size(max_uncompressed_chunk_bytes)
        if err != nvcompStatus_t.nvcompSuccess:
            raise RuntimeError(
                f"Could not get output buffer size, "
                f"error: {nvCompStatus(err)!r}."
            )
        return comp_chunk_size

    @abstractmethod
    def _get_comp_chunk_size(self, size_t max_uncompressed_chunk_bytes):
        """Algorithm-specific implementation."""
        ...

    def compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream,
    ):
        """Perform compression.

        Parameters
        ----------
        uncomp_chunks: cp.ndarray[uintp]
            The pointers on the GPU, to uncompressed batched items.
        uncomp_chunk_sizes: cp.ndarray[uint64]
            The size in bytes of each uncompressed batch item on the GPU.
        max_uncomp_chunk_bytes: int
            The maximum size in bytes of the largest chunk in the batch.
        batch_size: int
            The number of chunks to compress.
        temp_buf: cp.ndarray
            The temporary GPU workspace.
        comp_chunks: np.ndarray[uintp]
            (output) The list of pointers on the GPU, to the output location for each
            compressed batch item.
        comp_chunk_sizes: np.ndarray[uint64]
            (output) The compressed size in bytes of each chunk.
        stream: cp.cuda.Stream
            CUDA stream.
        """

        # nvCOMP requires comp_chunks pointers container and
        # comp_chunk_sizes to be in GPU memory.
        comp_chunks_d = cupy.array(comp_chunks, dtype=cupy.uintp)
        comp_chunk_sizes_d = cupy.empty_like(comp_chunk_sizes)

        err = self._compress(
            uncomp_chunks,
            uncomp_chunk_sizes,
            max_uncomp_chunk_bytes,
            batch_size,
            temp_buf,
            comp_chunks_d,
            comp_chunk_sizes_d,
            stream,
        )
        if err != nvcompStatus_t.nvcompSuccess:
            raise RuntimeError(f"Compression failed, error: {nvCompStatus(err)!r}.")
        # Copy resulting compressed chunk sizes back to the host buffer.
        comp_chunk_sizes[:] = comp_chunk_sizes_d.get()

    @abstractmethod
    def _compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream
    ):
        """Algorithm-specific implementation."""
        ...

    def get_decompress_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ):
        """Get the amount of temp space required on the GPU for decompression.

        Parameters
        ----------
        batch_size: int
            The number of items in the batch.
        max_uncompressed_chunk_bytes: int
            The size in bytes of the largest chunk when uncompressed.

        Returns
        -------
        int
            The amount of temporary GPU space in bytes that will be
            required to decompress.
        """
        err, temp_size = self._get_decomp_temp_size(
            batch_size,
            max_uncompressed_chunk_bytes
        )
        if err != nvcompStatus_t.nvcompSuccess:
            raise RuntimeError(
                f"Could not get decompress temp buffer size, "
                f"error: {nvCompStatus(err)!r}."
            )

        return temp_size

    @abstractmethod
    def _get_decomp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ):
        """Algorithm-specific implementation."""
        ...

    def get_decompress_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        stream,
    ):
        """Get the amount of space required on the GPU for decompression.

        Parameters
        ----------
        comp_chunks: np.ndarray[uintp]
            The pointers on the GPU, to compressed batched items.
        comp_chunk_sizes: np.ndarray[uint64]
            The size in bytes of each compressed batch item.
        stream: cp.cuda.Stream
            CUDA stream.

        Returns
        -------
        cp.ndarray[uint64]
            The amount of GPU space in bytes that will be required
            to decompress each chunk.
        """

        assert len(comp_chunks) == len(comp_chunk_sizes)
        batch_size = len(comp_chunks)

        # nvCOMP requires all buffers to be in GPU memory.
        comp_chunks_d = cupy.array(comp_chunks, dtype=cupy.uintp)
        comp_chunk_sizes_d = cupy.array(comp_chunk_sizes, dtype=cupy.uint64)
        uncomp_chunk_sizes_d = cupy.empty_like(comp_chunk_sizes_d)

        err = self._get_decomp_size(
            comp_chunks_d,
            comp_chunk_sizes_d,
            batch_size,
            uncomp_chunk_sizes_d,
            stream,
        )
        if err != nvcompStatus_t.nvcompSuccess:
            raise RuntimeError(
                f"Could not get decompress buffer size, error: {nvCompStatus(err)!r}."
            )

        return uncomp_chunk_sizes_d

    @abstractmethod
    def _get_decomp_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        uncomp_chunk_sizes,
        stream,
    ):
        """Algorithm-specific implementation."""
        ...

    def decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        """Perform decompression.

        Parameters
        ----------
        comp_chunks: np.ndarray[uintp]
            The pointers on the GPU, to compressed batched items.
        comp_chunk_sizes: np.ndarray[uint64]
            The size in bytes of each compressed batch item.
        batch_size: int
            The number of chunks to decompress.
        temp_buf: cp.ndarray
            The temporary GPU workspace.
        uncomp_chunks: cp.ndarray[uintp]
            (output) The pointers on the GPU, to the output location for each
            decompressed batch item.
        uncomp_chunk_sizes: cp.ndarray[uint64]
            The size in bytes of each decompress chunk location on the GPU.
        actual_uncomp_chunk_sizes: cp.ndarray[uint64]
            (output) The actual decompressed size in bytes of each chunk on the GPU.
        statuses: cp.ndarray
            (output) The status for each chunk of whether it was decompressed or not.
        stream: cp.cuda.Stream
            CUDA stream.
        """

        # nvCOMP requires comp_chunks pointers container and
        # comp_chunk_sizes to be in GPU memory.
        comp_chunks_d = cupy.array(comp_chunks, dtype=cupy.uintp)
        comp_chunk_sizes_d = cupy.array(comp_chunk_sizes, dtype=cupy.uint64)

        err = self._decompress(
            comp_chunks_d,
            comp_chunk_sizes_d,
            batch_size,
            temp_buf,
            uncomp_chunks,
            uncomp_chunk_sizes,
            actual_uncomp_chunk_sizes,
            statuses,
            stream,
        )
        if err != nvcompStatus_t.nvcompSuccess:
            raise RuntimeError(f"Decompression failed, error: {nvCompStatus(err)!r}.")

    @abstractmethod
    def _decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        """Algorithm-specific implementation."""
        ...


cdef uintptr_t to_ptr(buf):
    return buf.data.ptr


cdef cudaStream_t to_stream(stream):
    return <cudaStream_t><size_t>stream.ptr


#
# LZ4 algorithm.
#

from kvikio._lib.nvcomp_ll_cxx_api cimport (
    nvcompBatchedLZ4CompressAsync,
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
    nvcompBatchedLZ4CompressGetTempSize,
    nvcompBatchedLZ4DecompressAsync,
    nvcompBatchedLZ4DecompressGetTempSize,
    nvcompBatchedLZ4GetDecompressSizeAsync,
    nvcompBatchedLZ4Opts_t,
)


class nvCompBatchAlgorithmLZ4(nvCompBatchAlgorithm):
    """LZ4 algorithm implementation."""

    algo_id: str = "lz4"

    options: nvcompBatchedLZ4Opts_t

    HEADER_SIZE_BYTES: size_t = sizeof(uint32_t)

    def __init__(self, data_type: int = 0, has_header: bool = True):
        """Initialize the codec.

        Parameters
        ----------
        data_type: int
            Source data type.
        has_header: bool
            Whether the compressed data has a header.
            This enables data compatibility between numcodecs LZ4 codec,
            which has the header and nvCOMP LZ4 codec which does not
            require the header.
        """
        self.options = nvcompBatchedLZ4Opts_t(data_type)
        self.has_header = has_header

    def _get_comp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ) -> tuple[nvcompStatus_t, size_t]:
        cdef size_t temp_bytes = 0

        err = nvcompBatchedLZ4CompressGetTempSize(
            batch_size,
            max_uncompressed_chunk_bytes,
            self.options,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_comp_chunk_size(self, size_t max_uncompressed_chunk_bytes):
        cdef size_t max_compressed_bytes = 0

        err = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
            max_uncompressed_chunk_bytes,
            self.options,
            &max_compressed_bytes
        )

        # Add header size, if needed.
        if err == nvcompStatus_t.nvcompSuccess and self.has_header:
            max_compressed_bytes += self.HEADER_SIZE_BYTES

        return (err, max_compressed_bytes)

    def compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream,
    ):
        if self.has_header:
            # If there is a header, we need to:
            # 1. Copy the uncompressed chunk size to the compressed chunk header.
            # 2. Update target pointers in comp_chunks to skip the header portion,
            # which is not compressed.
            #
            # Get the base pointers to sizes.
            psize = to_ptr(uncomp_chunk_sizes)
            for i in range(batch_size):
                # Copy the original data size to the header.
                memcpyAsync(
                    <uintptr_t>comp_chunks[i],
                    psize,
                    self.HEADER_SIZE_BYTES,
                    cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream.ptr
                )
                psize += sizeof(uint64_t)
                # Update chunk pointer to skip the header.
                comp_chunks[i] += self.HEADER_SIZE_BYTES

        super().compress(
            uncomp_chunks,
            uncomp_chunk_sizes,
            max_uncomp_chunk_bytes,
            batch_size,
            temp_buf,
            comp_chunks,
            comp_chunk_sizes,
            stream,
        )

        if self.has_header:
            for i in range(batch_size):
                # Update chunk pointer and size to include the header.
                comp_chunks[i] -= self.HEADER_SIZE_BYTES
                comp_chunk_sizes[i] += self.HEADER_SIZE_BYTES

    def _compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream
    ):
        # Cast buffer pointers that have Python int type to appropriate C types
        # suitable for passing to nvCOMP API.
        return nvcompBatchedLZ4CompressAsync(
            <const void* const*>to_ptr(uncomp_chunks),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            max_uncomp_chunk_bytes,
            batch_size,
            <void*>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(comp_chunks),
            <size_t*>to_ptr(comp_chunk_sizes),
            self.options,
            to_stream(stream),
        )

    def _get_decomp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ):
        cdef size_t temp_bytes = 0

        err = nvcompBatchedLZ4DecompressGetTempSize(
            batch_size,
            max_uncompressed_chunk_bytes,
            &temp_bytes
        )

        return (err, temp_bytes)

    def get_decompress_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        stream,
    ):
        if not self.has_header:
            return super().get_decompress_size(
                comp_chunks,
                comp_chunk_sizes,
                stream,
            )

        assert comp_chunks.shape == comp_chunk_sizes.shape
        batch_size = len(comp_chunks)

        # uncomp_chunk_sizes is uint32 array to match the type in LZ4 header.
        uncomp_chunk_sizes = cupy.empty(batch_size, dtype=cupy.uint32)

        psize = to_ptr(uncomp_chunk_sizes)
        for i in range(batch_size):
            # Get pointer to the header and copy the data.
            memcpyAsync(
                psize,
                <uintptr_t>comp_chunks[i],
                sizeof(uint32_t),
                cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.ptr
            )
            psize += sizeof(uint32_t)
        stream.synchronize()

        return uncomp_chunk_sizes.astype(cupy.uint64)

    def _get_decomp_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        uncomp_chunk_sizes,
        stream,
    ):
        return nvcompBatchedLZ4GetDecompressSizeAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <size_t*>to_ptr(uncomp_chunk_sizes),
            batch_size,
            to_stream(stream),
        )

    def decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        if self.has_header:
            for i in range(batch_size):
                # Update chunk pointer and size to exclude the header.
                comp_chunks[i] += self.HEADER_SIZE_BYTES
                comp_chunk_sizes[i] -= self.HEADER_SIZE_BYTES

        super().decompress(
            comp_chunks,
            comp_chunk_sizes,
            batch_size,
            temp_buf,
            uncomp_chunks,
            uncomp_chunk_sizes,
            actual_uncomp_chunk_sizes,
            statuses,
            stream,
        )

        if self.has_header:
            for i in range(batch_size):
                # Update chunk pointer and size to include the header.
                comp_chunks[i] -= self.HEADER_SIZE_BYTES
                comp_chunk_sizes[i] += self.HEADER_SIZE_BYTES

    def _decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        # Cast buffer pointers that have Python int type to appropriate C types
        # suitable for passing to nvCOMP API.
        return nvcompBatchedLZ4DecompressAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            <size_t*>NULL,
            batch_size,
            <void* const>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(uncomp_chunks),
            <nvcompStatus_t*>NULL,
            to_stream(stream),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(data_type={self.options['data_type']})"


#
# Gdeflate algorithm.
#
from kvikio._lib.nvcomp_ll_cxx_api cimport (
    nvcompBatchedGdeflateCompressAsync,
    nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
    nvcompBatchedGdeflateCompressGetTempSize,
    nvcompBatchedGdeflateDecompressAsync,
    nvcompBatchedGdeflateDecompressGetTempSize,
    nvcompBatchedGdeflateGetDecompressSizeAsync,
    nvcompBatchedGdeflateOpts_t,
)


class nvCompBatchAlgorithmGdeflate(nvCompBatchAlgorithm):
    """Gdeflate algorithm implementation."""

    algo_id: str = "gdeflate"

    options: nvcompBatchedGdeflateOpts_t

    def __init__(self, algo: int = 0):
        self.options = nvcompBatchedGdeflateOpts_t(algo)

    def _get_comp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ) -> tuple[nvcompStatus_t, size_t]:
        cdef size_t temp_bytes = 0

        err = nvcompBatchedGdeflateCompressGetTempSize(
            batch_size,
            max_uncompressed_chunk_bytes,
            self.options,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_comp_chunk_size(self, size_t max_uncompressed_chunk_bytes):
        cdef size_t max_compressed_bytes = 0

        err = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
            max_uncompressed_chunk_bytes,
            self.options,
            &max_compressed_bytes
        )

        return (err, max_compressed_bytes)

    def _compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream
    ):
        return nvcompBatchedGdeflateCompressAsync(
            <const void* const*>to_ptr(uncomp_chunks),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            max_uncomp_chunk_bytes,
            batch_size,
            <void*>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(comp_chunks),
            <size_t*>to_ptr(comp_chunk_sizes),
            self.options,
            to_stream(stream),
        )

    def _get_decomp_temp_size(
        self,
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
    ):
        cdef size_t temp_bytes = 0

        err = nvcompBatchedGdeflateDecompressGetTempSize(
            num_chunks,
            max_uncompressed_chunk_bytes,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_decomp_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        uncomp_chunk_sizes,
        stream,
    ):
        return nvcompBatchedGdeflateGetDecompressSizeAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <size_t*>to_ptr(uncomp_chunk_sizes),
            batch_size,
            to_stream(stream),
        )

    def _decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        return nvcompBatchedGdeflateDecompressAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            <size_t*>NULL,
            batch_size,
            <void* const>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(uncomp_chunks),
            <nvcompStatus_t*>NULL,
            to_stream(stream),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(algo={self.options['algo']})"


#
# zstd algorithm.
#
from kvikio._lib.nvcomp_ll_cxx_api cimport (
    nvcompBatchedZstdCompressAsync,
    nvcompBatchedZstdCompressGetMaxOutputChunkSize,
    nvcompBatchedZstdCompressGetTempSize,
    nvcompBatchedZstdDecompressAsync,
    nvcompBatchedZstdDecompressGetTempSize,
    nvcompBatchedZstdGetDecompressSizeAsync,
    nvcompBatchedZstdOpts_t,
)


class nvCompBatchAlgorithmZstd(nvCompBatchAlgorithm):
    """zstd algorithm implementation."""

    algo_id: str = "zstd"

    options: nvcompBatchedZstdOpts_t

    def __init__(self):
        self.options = nvcompBatchedZstdOpts_t(0)

    def _get_comp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ) -> tuple[nvcompStatus_t, size_t]:
        cdef size_t temp_bytes = 0

        err = nvcompBatchedZstdCompressGetTempSize(
            batch_size,
            max_uncompressed_chunk_bytes,
            self.options,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_comp_chunk_size(self, size_t max_uncompressed_chunk_bytes):
        cdef size_t max_compressed_bytes = 0

        err = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            max_uncompressed_chunk_bytes,
            self.options,
            &max_compressed_bytes
        )

        return (err, max_compressed_bytes)

    def _compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream
    ):
        return nvcompBatchedZstdCompressAsync(
            <const void* const*>to_ptr(uncomp_chunks),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            max_uncomp_chunk_bytes,
            batch_size,
            <void*>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(comp_chunks),
            <size_t*>to_ptr(comp_chunk_sizes),
            self.options,
            to_stream(stream),
        )

    def _get_decomp_temp_size(
        self,
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
    ):
        cdef size_t temp_bytes = 0

        err = nvcompBatchedZstdDecompressGetTempSize(
            num_chunks,
            max_uncompressed_chunk_bytes,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_decomp_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        uncomp_chunk_sizes,
        stream,
    ):
        return nvcompBatchedZstdGetDecompressSizeAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <size_t*>to_ptr(uncomp_chunk_sizes),
            batch_size,
            to_stream(stream),
        )

    def _decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        return nvcompBatchedZstdDecompressAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            <size_t*>to_ptr(actual_uncomp_chunk_sizes),
            batch_size,
            <void* const>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(uncomp_chunks),
            <nvcompStatus_t*>to_ptr(statuses),
            to_stream(stream),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"


#
# Snappy algorithm.
#
from kvikio._lib.nvcomp_ll_cxx_api cimport (
    nvcompBatchedSnappyCompressAsync,
    nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
    nvcompBatchedSnappyCompressGetTempSize,
    nvcompBatchedSnappyDecompressAsync,
    nvcompBatchedSnappyDecompressGetTempSize,
    nvcompBatchedSnappyGetDecompressSizeAsync,
    nvcompBatchedSnappyOpts_t,
)


class nvCompBatchAlgorithmSnappy(nvCompBatchAlgorithm):
    """Snappy algorithm implementation."""

    algo_id: str = "snappy"

    options: nvcompBatchedSnappyOpts_t

    def __init__(self):
        self.options = nvcompBatchedSnappyOpts_t(0)

    def _get_comp_temp_size(
        self,
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
    ) -> tuple[nvcompStatus_t, size_t]:
        cdef size_t temp_bytes = 0

        err = nvcompBatchedSnappyCompressGetTempSize(
            batch_size,
            max_uncompressed_chunk_bytes,
            self.options,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_comp_chunk_size(self, size_t max_uncompressed_chunk_bytes):
        cdef size_t max_compressed_bytes = 0

        err = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
            max_uncompressed_chunk_bytes,
            self.options,
            &max_compressed_bytes
        )

        return (err, max_compressed_bytes)

    def _compress(
        self,
        uncomp_chunks,
        uncomp_chunk_sizes,
        size_t max_uncomp_chunk_bytes,
        size_t batch_size,
        temp_buf,
        comp_chunks,
        comp_chunk_sizes,
        stream
    ):
        return nvcompBatchedSnappyCompressAsync(
            <const void* const*>to_ptr(uncomp_chunks),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            max_uncomp_chunk_bytes,
            batch_size,
            <void*>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(comp_chunks),
            <size_t*>to_ptr(comp_chunk_sizes),
            self.options,
            to_stream(stream),
        )

    def _get_decomp_temp_size(
        self,
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
    ):
        cdef size_t temp_bytes = 0

        err = nvcompBatchedSnappyDecompressGetTempSize(
            num_chunks,
            max_uncompressed_chunk_bytes,
            &temp_bytes
        )

        return (err, temp_bytes)

    def _get_decomp_size(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        uncomp_chunk_sizes,
        stream,
    ):
        return nvcompBatchedSnappyGetDecompressSizeAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <size_t*>to_ptr(uncomp_chunk_sizes),
            batch_size,
            to_stream(stream),
        )

    def _decompress(
        self,
        comp_chunks,
        comp_chunk_sizes,
        size_t batch_size,
        temp_buf,
        uncomp_chunks,
        uncomp_chunk_sizes,
        actual_uncomp_chunk_sizes,
        statuses,
        stream,
    ):
        return nvcompBatchedSnappyDecompressAsync(
            <const void* const*>to_ptr(comp_chunks),
            <const size_t*>to_ptr(comp_chunk_sizes),
            <const size_t*>to_ptr(uncomp_chunk_sizes),
            <size_t*>NULL,
            batch_size,
            <void* const>to_ptr(temp_buf),
            <size_t>temp_buf.nbytes,
            <void* const*>to_ptr(uncomp_chunks),
            <nvcompStatus_t*>NULL,
            to_stream(stream),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"


SUPPORTED_ALGORITHMS = {
    a.algo_id: a for a in [
        nvCompBatchAlgorithmLZ4,
        nvCompBatchAlgorithmGdeflate,
        nvCompBatchAlgorithmZstd,
        nvCompBatchAlgorithmSnappy,
    ]
}
