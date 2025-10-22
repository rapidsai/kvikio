# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kvikio


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(kvikio.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(kvikio.__version__, str)
    assert len(kvikio.__version__) > 0
