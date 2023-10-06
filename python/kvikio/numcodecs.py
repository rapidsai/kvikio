# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

"""
This module implements CUDA compression and transformation codecs for Numcodecs.
See <https://numcodecs.readthedocs.io/en/stable/>
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Union

import cupy.typing
import numpy.typing
from numcodecs.abc import Codec

# TODO: replace `ANY` with `collections.abc.Buffer` from PEP-688
# when it becomes available.
BufferLike = Union[cupy.typing.NDArray, numpy.typing.ArrayLike, Any]


class CudaCodec(Codec):
    """Abstract base class for CUDA codecs"""

    @abstractmethod
    def encode(self, buf: BufferLike) -> cupy.typing.NDArray:
        """Encode `buf` using CUDA.

        This method should support both device and host buffers.

        Parameters
        ----------
        buf
            A numpy array like object such as numpy.ndarray, cupy.ndarray,
            or any object exporting a buffer interface.

        Returns
        -------
        The compressed buffer wrapped in a CuPy array
        """

    @abstractmethod
    def decode(self, buf: BufferLike, out: Optional[BufferLike] = None) -> BufferLike:
        """Decode `buf` using CUDA.

        This method should support both device and host buffers.

        Parameters
        ----------
        buf
            A numpy array like object such as numpy.ndarray, cupy.ndarray,
            or any object exporting a buffer interface.
        out
            A numpy array like object such as numpy.ndarray, cupy.ndarray,
            or any object exporting a buffer interface. If provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
            Decoded data, which is either host or device memory based on the type
            of `out`. If `out` is None, the type of `buf` determines the return buffer
            type.
        """
