# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from kvikio._nvcomp_codec import NvCompBatchCodec  # noqa: F401
from kvikio.utils import kvikio_deprecate_module

kvikio_deprecate_module(
    "Use the official nvCOMP API from 'nvidia.nvcomp' instead.", since="25.06"
)
