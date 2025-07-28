# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from kvikio._nvcomp import (  # noqa: F401
    ANSManager,
    BitcompManager,
    CascadedManager,
    GdeflateManager,
    LZ4Manager,
    ManagedDecompressionManager,
    SnappyManager,
    cp_to_nvcomp_dtype,
    nvCompManager,
)
from kvikio.utils import kvikio_deprecate_module

kvikio_deprecate_module(
    "Use the official nvCOMP API from 'nvidia.nvcomp' instead.", since="25.06"
)
