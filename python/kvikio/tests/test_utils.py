# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

import kvikio.utils


def test_remote_file_rejects_invalid_internal_handle(monkeypatch):
    import kvikio.remote_file

    class RemoteHandle:
        pass

    monkeypatch.setattr(
        kvikio.remote_file,
        "_get_remote_module",
        lambda: type("RemoteModule", (), {"RemoteFile": RemoteHandle}),
    )

    with pytest.raises(TypeError, match="handle must be a RemoteFile instance"):
        kvikio.remote_file.RemoteFile(object())


def test_function_deprecation():
    """Test the decorator used to deprecate functions"""

    @kvikio.utils.kvikio_deprecation_notice("my deprecation notice", since="my_version")
    def func():
        pass

    with pytest.warns(FutureWarning, match="my deprecation notice"):
        func()

    assert ".. deprecated:: my_version" in func.__doc__


def test_module_deprecation():
    """Test the utility function used to deprecate modules"""

    with pytest.warns(FutureWarning, match="my_version.*my deprecation notice"):
        kvikio.utils.kvikio_deprecate_module(
            "my deprecation notice", since="my_version"
        )
