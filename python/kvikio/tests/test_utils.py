# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

import kvikio.utils


def test_function_deprecation():
    """Test the decorator used to deprecate functions"""

    @kvikio.utils.kvikio_deprecation_notice("my deprecation notice", since="my_version")
    def func():
        pass

    with pytest.warns(FutureWarning, match="my deprecation notice"):
        func()

    assert ".. deprecated:: my_version" in getattr(func, "__doc__")


def test_module_deprecation():
    """Test the utility function used to deprecate modules"""

    with pytest.warns(FutureWarning, match="my_version.*my deprecation notice"):
        kvikio.utils.kvikio_deprecate_module(
            "my deprecation notice", since="my_version"
        )
