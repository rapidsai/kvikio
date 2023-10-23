# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from typing import Any, Mapping, Optional, Sequence

import cupy as cp
import cupy.typing
from numcodecs.compat import ensure_contiguous_ndarray_like

from kvikio._lib.libnvcomp_ll import SUPPORTED_ALGORITHMS
from kvikio.numcodecs import BufferLike, CudaCodec


class NvCompBatchCodec(CudaCodec):
    """Codec that uses batch algorithms from nvCOMP library.

    An algorithm is selected using `algorithm` parameter.
    If the algorithm takes additional options, they can be
    passed to the algorithm using `options` dictionary.
    """

    # Header stores original uncompressed size. This is required to enable
    # data compatibility between existing numcodecs codecs and NvCompBatchCodec.
    HEADER_SIZE_BYTES: int = 4

    codec_id: str = "nvcomp_batch"
    algorithm: str
    options: Mapping[str, Any]

    def __init__(
        self,
        algorithm: str,
        options: Optional[Mapping[str, Any]] = None,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        algo_id = algorithm.lower()
        algo_t = SUPPORTED_ALGORITHMS.get(algo_id, None)
        if algo_t is None:
            raise ValueError(
                f"{algorithm} is not supported. "
                f"Must be one of: {list(SUPPORTED_ALGORITHMS.keys())}"
            )

        self.algorithm = algo_id
        self.options = dict(options) if options is not None else {}

        # Create an algorithm.
        self._algo = algo_t(**self.options)
        # Use default stream, if needed.
        self._stream = stream if stream is not None else cp.cuda.Stream.ptds

    def encode(self, buf: BufferLike) -> cupy.typing.NDArray:
        return self.encode_batch([buf])[0]

    def encode_batch(self, bufs: Sequence[Any]) -> Sequence[Any]:
        """Encode data in `bufs` using nvCOMP.

        Parameters
        ----------
        bufs :
            Data to be encoded. Each buffer in the list may be any object
            supporting the new-style buffer protocol.

        Returns
        -------
            List of encoded buffers. Each buffer may be any object supporting
            the new-style buffer protocol.
        """
        num_chunks = len(bufs)
        if num_chunks == 0:
            return []

        bufs = [cp.asarray(ensure_contiguous_ndarray_like(b)) for b in bufs]
        buf_sizes = [b.size * b.itemsize for b in bufs]

        max_chunk_size = max(buf_sizes)

        # Get temp and output buffer sizes.
        temp_size = self._algo.get_compress_temp_size(num_chunks, max_chunk_size)
        comp_chunk_size = self._algo.get_compress_chunk_size(max_chunk_size)

        # Prepare data and size buffers.
        # uncomp_chunks is used as a container that stores pointers to actual chunks.
        # nvCOMP requires this and sizes buffers to be in GPU memory.
        uncomp_chunks = cp.array([b.data.ptr for b in bufs], dtype=cp.uintp)
        uncomp_chunk_sizes = cp.array(buf_sizes, dtype=cp.uint64)

        temp_buf = cp.empty(temp_size, dtype=cp.uint8)

        comp_chunks = cp.empty((num_chunks, comp_chunk_size), dtype=cp.uint8)
        # Array of pointers to each compressed chunk.
        comp_chunk_ptrs = cp.array([c.data.ptr for c in comp_chunks], dtype=cp.uintp)
        # Resulting compressed chunk sizes.
        comp_chunk_sizes = cp.empty(num_chunks, dtype=cp.uint64)

        self._algo.compress(
            uncomp_chunks,
            uncomp_chunk_sizes,
            max_chunk_size,
            num_chunks,
            temp_buf,
            comp_chunk_ptrs,
            comp_chunk_sizes,
            self._stream,
        )

        res = []
        # Copy to host to subsequently avoid many smaller D2H copies.
        comp_chunks = cp.asnumpy(comp_chunks, self._stream)
        comp_chunk_sizes = cp.asnumpy(comp_chunk_sizes, self._stream)
        self._stream.synchronize()

        for i in range(num_chunks):
            res.append(comp_chunks[i, : comp_chunk_sizes[i]].tobytes())
        return res

    def decode(self, buf: BufferLike, out: Optional[BufferLike] = None) -> BufferLike:
        return self.decode_batch([buf], [out])[0]

    def decode_batch(
        self, bufs: Sequence[Any], out: Optional[Sequence[Any]] = None
    ) -> Sequence[Any]:
        """Decode data in `bufs` using nvCOMP.

        Parameters
        ----------
        bufs :
            Encoded data. Each buffer in the list may be any object
            supporting the new-style buffer protocol.
        out :
            List of writeable buffers to store decoded data.
            N.B. if provided, each buffer must be exactly the right size
            to store the decoded data.

        Returns
        -------
            List of decoded buffers. Each buffer may be any object supporting
            the new-style buffer protocol.
        """
        num_chunks = len(bufs)
        if num_chunks == 0:
            return []

        # TODO(akamenev): check only first buffer, assuming they are all
        # of the same kind.
        is_host_buffer = not hasattr(bufs[0], "__cuda_array_interface__")
        if is_host_buffer:
            bufs = [cp.asarray(ensure_contiguous_ndarray_like(b)) for b in bufs]

        # Prepare compressed chunks buffers.
        comp_chunks = cp.array([b.data.ptr for b in bufs], dtype=cp.uintp)
        comp_chunk_sizes = cp.array([b.size for b in bufs], dtype=cp.uint64)

        # Get uncompressed chunk sizes.
        uncomp_chunk_sizes = self._algo.get_decompress_size(
            comp_chunks,
            comp_chunk_sizes,
            self._stream,
        )

        # Check whether the uncompressed chunks are all the same size.
        # cupy.unique returns sorted sizes.
        sorted_chunk_sizes = cp.unique(uncomp_chunk_sizes)
        max_chunk_size = sorted_chunk_sizes[-1].item()
        is_equal_chunks = sorted_chunk_sizes.shape[0] == 1

        # Get temp buffer size.
        temp_size = self._algo.get_decompress_temp_size(num_chunks, max_chunk_size)

        temp_buf = cp.empty(temp_size, dtype=cp.uint8)

        # Prepare uncompressed chunks buffers.
        # First, allocate chunks of max_chunk_size and then
        # copy the pointers to a pointer array in GPU memory as required by nvCOMP.
        # For performance reasons, we use max_chunk_size so we can create
        # a rectangular array with the same pointer increments.
        uncomp_chunks = cp.empty((num_chunks, max_chunk_size), dtype=cp.uint8)
        p_start = uncomp_chunks.data.ptr
        uncomp_chunk_ptrs = cp.uint64(p_start) + (
            cp.arange(0, num_chunks * max_chunk_size, max_chunk_size, dtype=cp.uint64)
        )

        # TODO(akamenev): currently we provide the following 2 buffers to decompress()
        # but do not check/use them afterwards since some of the algos
        # (e.g. LZ4 and Gdeflate) do not require it and run faster
        # without those arguments passed, while other algos (e.g. zstd) require
        # these buffers to be valid.
        actual_uncomp_chunk_sizes = cp.empty(num_chunks, dtype=cp.uint64)
        statuses = cp.empty(num_chunks, dtype=cp.int32)

        self._algo.decompress(
            comp_chunks,
            comp_chunk_sizes,
            num_chunks,
            temp_buf,
            uncomp_chunk_ptrs,
            uncomp_chunk_sizes,
            actual_uncomp_chunk_sizes,
            statuses,
            self._stream,
        )

        # If all chunks are the same size, we can just return uncomp_chunks.
        if is_equal_chunks and out is None:
            return cp.asnumpy(uncomp_chunks) if is_host_buffer else uncomp_chunks

        res = []
        uncomp_chunk_sizes = uncomp_chunk_sizes.get()
        for i in range(num_chunks):
            ret = uncomp_chunks[i, : uncomp_chunk_sizes[i]]
            if out is None or out[i] is None:
                res.append(cp.asnumpy(ret) if is_host_buffer else ret)
            else:
                o = ensure_contiguous_ndarray_like(out[i])
                if hasattr(o, "__cuda_array_interface__"):
                    cp.copyto(o, ret.view(dtype=o.dtype), casting="no")
                else:
                    cp.asnumpy(ret.view(dtype=o.dtype), out=o, stream=self._stream)
                res.append(o)
        self._stream.synchronize()

        return res

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(algorithm={self.algorithm!r}, options={self.options!r})"
        )
