# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from typing import Any, List, Mapping, Optional

import cupy as cp
import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray_like

import kvikio._lib.libnvcomp_ll as _ll


class NvCompBatchCodec(Codec):
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
        algo_t = _ll.SUPPORTED_ALGORITHMS.get(algo_id, None)
        if algo_t is None:
            raise ValueError(
                f"{algorithm} is not supported. "
                f"Must be one of: {list(_ll.SUPPORTED_ALGORITHMS.keys())}"
            )

        self.algorithm = algo_id
        self.options = dict(options) if options is not None else {}

        # Create an algorithm.
        self._algo = algo_t(**self.options)
        # Use default stream, if needed.
        self._stream = stream if stream is not None else cp.cuda.Stream.ptds

    def encode(self, buf):
        """Encode data in `buf` using nvCOMP.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """
        return self.encode_batch([buf])[0]

    def encode_batch(self, bufs: List[Any]) -> List[Any]:
        """Encode data in `bufs` using nvCOMP.

        Parameters
        ----------
        bufs : List[buffer-like].
            Data to be encoded. Each buffer in the list may be any object
            supporting the new-style buffer protocol.

        Returns
        -------
        enc : List[buffer-like]
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
        comp_chunk_ptrs = np.array([c.data.ptr for c in comp_chunks], dtype=cp.uintp)
        # Resulting compressed chunk sizes.
        comp_chunk_sizes = np.empty(num_chunks, dtype=np.uint64)

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
        for i in range(num_chunks):
            res.append(comp_chunks[i, : comp_chunk_sizes[i]].tobytes())
        return res

    def decode(self, buf, out=None):
        """Decode data in `buf` using nvCOMP.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : buffer-like, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : buffer-like
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """
        return self.decode_batch([buf], [out])[0]

    def decode_batch(
        self, bufs: List[Any], out: Optional[List[Any]] = None
    ) -> List[Any]:
        """Decode data in `bufs` using nvCOMP.

        Parameters
        ----------
        bufs : List[buffer-like]
            Encoded data. Each buffer in the list may be any object
            supporting the new-style buffer protocol.
        out : List[buffer-like], optional
            List of writeable buffers to store decoded data.
            N.B. if provided, each buffer must be exactly the right size
            to store the decoded data.

        Returns
        -------
        dec : List[buffer-like]
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
        comp_chunks = np.array([b.data.ptr for b in bufs], dtype=np.uintp)
        comp_chunk_sizes = np.array([b.size for b in bufs], dtype=np.uint64)

        # Get uncompressed chunk sizes.
        uncomp_chunk_sizes = self._algo.get_decompress_size(
            comp_chunks,
            comp_chunk_sizes,
            self._stream,
        )
        # Copy to host since we'll need it to properly allocate buffers.
        uncomp_chunk_sizes_h = uncomp_chunk_sizes.get()

        max_chunk_size = uncomp_chunk_sizes_h.max()

        # Get temp buffer size.
        temp_size = self._algo.get_decompress_temp_size(num_chunks, max_chunk_size)

        temp_buf = cp.empty(temp_size, dtype=cp.uint8)

        # Prepare uncompressed chunks buffers.
        # First, allocate chunks of appropriate sizes and then
        # copy the pointers to a pointer array in GPU memory as required by nvCOMP.
        # TODO(akamenev): probably can allocate single contiguous buffer.
        uncomp_chunks = [
            cp.empty(size, dtype=cp.uint8) for size in uncomp_chunk_sizes_h
        ]
        uncomp_chunk_ptrs = cp.array(
            [c.data.ptr for c in uncomp_chunks], dtype=cp.uintp
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

        res = []
        for i in range(num_chunks):
            ret = uncomp_chunks[i]
            if out is not None and out[i] is not None:
                o = ensure_contiguous_ndarray_like(out[i])
                if hasattr(o, "__cuda_array_interface__"):
                    cp.copyto(o, ret.view(dtype=o.dtype), casting="no")
                else:
                    np.copyto(o, cp.asnumpy(ret.view(dtype=o.dtype)), casting="no")
                res.append(o)
            elif is_host_buffer:
                res.append(cp.asnumpy(ret))
            else:
                res.append(ret)

        return res

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(algorithm={self.algorithm!r}, options={self.options!r})"
        )
