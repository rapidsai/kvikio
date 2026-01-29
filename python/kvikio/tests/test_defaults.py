# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

import kvikio.defaults


def test_property_accessor():
    """Test the method `get` and `set`"""

    # Attempt to set a nonexistent property
    with pytest.raises(KeyError):
        kvikio.defaults.set("nonexistent_property", 123)

    # Attempt to get a nonexistent property
    with pytest.raises(KeyError):
        kvikio.defaults.get("nonexistent_property")

    # Attempt to set a property whose name is mistakenly prefixed by "set_"
    # (coinciding with the setter method).
    with pytest.raises(KeyError):
        kvikio.defaults.set("set_task_size", 123)

    # Nested context managers
    task_size_default = kvikio.defaults.get("task_size")
    with kvikio.defaults.set("task_size", 1024):
        assert kvikio.defaults.get("task_size") == 1024
        with kvikio.defaults.set("task_size", 2048):
            assert kvikio.defaults.get("task_size") == 2048
            with kvikio.defaults.set("task_size", 4096):
                assert kvikio.defaults.get("task_size") == 4096
            assert kvikio.defaults.get("task_size") == 2048
        assert kvikio.defaults.get("task_size") == 1024
    assert kvikio.defaults.get("task_size") == task_size_default

    # Multiple context managers
    task_size_default = kvikio.defaults.get("task_size")
    num_threads_default = kvikio.defaults.get("num_threads")
    bounce_buffer_size_default = kvikio.defaults.get("bounce_buffer_size")
    with kvikio.defaults.set(
        {"task_size": 1024, "num_threads": 16, "bounce_buffer_size": 1024}
    ):
        assert (
            (kvikio.defaults.get("task_size") == 1024)
            and (kvikio.defaults.get("num_threads") == 16)
            and (kvikio.defaults.get("bounce_buffer_size") == 1024)
        )
    assert (
        (kvikio.defaults.get("task_size") == task_size_default)
        and (kvikio.defaults.get("num_threads") == num_threads_default)
        and (kvikio.defaults.get("bounce_buffer_size") == bounce_buffer_size_default)
    )


@pytest.mark.skipif(
    kvikio.defaults.get("compat_mode") == kvikio.CompatMode.ON,
    reason="cannot test `compat_mode` when already running in compatibility mode",
)
def test_compat_mode():
    """Test changing `compat_mode`"""

    before = kvikio.defaults.get("compat_mode")
    with kvikio.defaults.set("compat_mode", kvikio.CompatMode.ON):
        assert kvikio.defaults.get("compat_mode") == kvikio.CompatMode.ON
        kvikio.defaults.set("compat_mode", kvikio.CompatMode.OFF)
        assert kvikio.defaults.get("compat_mode") == kvikio.CompatMode.OFF
        kvikio.defaults.set("compat_mode", kvikio.CompatMode.AUTO)
        assert kvikio.defaults.get("compat_mode") == kvikio.CompatMode.AUTO
    assert before == kvikio.defaults.get("compat_mode")


def test_num_threads():
    """Test changing `num_threads`"""

    before = kvikio.defaults.get("num_threads")
    with kvikio.defaults.set("num_threads", 3):
        assert kvikio.defaults.get("num_threads") == 3
        kvikio.defaults.set("num_threads", 4)
        assert kvikio.defaults.get("num_threads") == 4
    assert before == kvikio.defaults.get("num_threads")

    with pytest.raises(ValueError, match="positive integer"):
        kvikio.defaults.set("num_threads", 0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.set("num_threads", -1)


def test_task_size():
    """Test changing `task_size`"""

    before = kvikio.defaults.get("task_size")
    with kvikio.defaults.set("task_size", 3):
        assert kvikio.defaults.get("task_size") == 3
        kvikio.defaults.set("task_size", 4)
        assert kvikio.defaults.get("task_size") == 4
    assert before == kvikio.defaults.get("task_size")

    with pytest.raises(ValueError, match="positive integer"):
        kvikio.defaults.set("task_size", 0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.set("task_size", -1)


def test_gds_threshold():
    """Test changing `gds_threshold`"""

    before = kvikio.defaults.get("gds_threshold")
    with kvikio.defaults.set("gds_threshold", 3):
        assert kvikio.defaults.get("gds_threshold") == 3
        kvikio.defaults.set("gds_threshold", 4)
        assert kvikio.defaults.get("gds_threshold") == 4
    assert before == kvikio.defaults.get("gds_threshold")

    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.set("gds_threshold", -1)


def test_bounce_buffer_size():
    """Test changing `bounce_buffer_size`"""

    before = kvikio.defaults.get("bounce_buffer_size")
    with kvikio.defaults.set("bounce_buffer_size", 3):
        assert kvikio.defaults.get("bounce_buffer_size") == 3
        kvikio.defaults.set("bounce_buffer_size", 4)
        assert kvikio.defaults.get("bounce_buffer_size") == 4
    assert before == kvikio.defaults.get("bounce_buffer_size")

    with pytest.raises(ValueError, match="positive integer"):
        kvikio.defaults.set("bounce_buffer_size", 0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.set("bounce_buffer_size", -1)


def test_http_max_attempts():
    before = kvikio.defaults.get("http_max_attempts")

    with kvikio.defaults.set("http_max_attempts", 5):
        assert kvikio.defaults.get("http_max_attempts") == 5
        kvikio.defaults.set("http_max_attempts", 4)
        assert kvikio.defaults.get("http_max_attempts") == 4
    assert kvikio.defaults.get("http_max_attempts") == before

    with pytest.raises(ValueError, match="positive integer"):
        kvikio.defaults.set("http_max_attempts", 0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.set("http_max_attempts", -1)


def test_http_status_codes():
    before = kvikio.defaults.get("http_status_codes")

    with kvikio.defaults.set("http_status_codes", [500]):
        assert kvikio.defaults.get("http_status_codes") == [500]
        kvikio.defaults.set("http_status_codes", [429, 500])
        assert kvikio.defaults.get("http_status_codes") == [429, 500]
    assert kvikio.defaults.get("http_status_codes") == before

    with pytest.raises(TypeError):
        kvikio.defaults.set("http_status_codes", 0)

    with pytest.raises(TypeError):
        kvikio.defaults.set("http_status_codes", ["a"])


def test_http_timeout():
    before = kvikio.defaults.get("http_timeout")

    with kvikio.defaults.set("http_timeout", 1):
        assert kvikio.defaults.get("http_timeout") == 1
    assert kvikio.defaults.get("http_timeout") == before
