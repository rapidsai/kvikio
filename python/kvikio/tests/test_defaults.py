# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

import kvikio.defaults


@pytest.mark.skipif(
    kvikio.defaults.compat_mode() == kvikio.CompatMode.ON,
    reason="cannot test `compat_mode` when already running in compatibility mode",
)
def test_compat_mode():
    """Test changing `compat_mode`"""

    before = kvikio.defaults.compat_mode()
    with kvikio.defaults.set_compat_mode(kvikio.CompatMode.ON):
        assert kvikio.defaults.compat_mode() == kvikio.CompatMode.ON
        kvikio.defaults.compat_mode_reset(kvikio.CompatMode.OFF)
        assert kvikio.defaults.compat_mode() == kvikio.CompatMode.OFF
        kvikio.defaults.compat_mode_reset(kvikio.CompatMode.AUTO)
        assert kvikio.defaults.compat_mode() == kvikio.CompatMode.AUTO
    assert before == kvikio.defaults.compat_mode()


def test_num_threads():
    """Test changing `num_threads`"""

    before = kvikio.defaults.get_num_threads()
    with kvikio.defaults.set_num_threads(3):
        assert kvikio.defaults.get_num_threads() == 3
        kvikio.defaults.num_threads_reset(4)
        assert kvikio.defaults.get_num_threads() == 4
    assert before == kvikio.defaults.get_num_threads()

    with pytest.raises(ValueError, match="positive integer greater than zero"):
        kvikio.defaults.num_threads_reset(0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.num_threads_reset(-1)


def test_task_size():
    """Test changing `task_size`"""

    before = kvikio.defaults.task_size()
    with kvikio.defaults.set_task_size(3):
        assert kvikio.defaults.task_size() == 3
        kvikio.defaults.task_size_reset(4)
        assert kvikio.defaults.task_size() == 4
    assert before == kvikio.defaults.task_size()

    with pytest.raises(ValueError, match="positive integer greater than zero"):
        kvikio.defaults.task_size_reset(0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.task_size_reset(-1)


def test_gds_threshold():
    """Test changing `gds_threshold`"""

    before = kvikio.defaults.gds_threshold()
    with kvikio.defaults.set_gds_threshold(3):
        assert kvikio.defaults.gds_threshold() == 3
        kvikio.defaults.gds_threshold_reset(4)
        assert kvikio.defaults.gds_threshold() == 4
    assert before == kvikio.defaults.gds_threshold()

    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.gds_threshold_reset(-1)


def test_bounce_buffer_size():
    """Test changing `bounce_buffer_size`"""

    before = kvikio.defaults.bounce_buffer_size()
    with kvikio.defaults.set_bounce_buffer_size(3):
        assert kvikio.defaults.bounce_buffer_size() == 3
        kvikio.defaults.bounce_buffer_size_reset(4)
        assert kvikio.defaults.bounce_buffer_size() == 4
    assert before == kvikio.defaults.bounce_buffer_size()

    with pytest.raises(ValueError, match="positive integer greater than zero"):
        kvikio.defaults.bounce_buffer_size_reset(0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.bounce_buffer_size_reset(-1)


def test_max_attempts():
    before = kvikio.defaults.max_attempts()

    with kvikio.defaults.set_max_attempts(5):
        assert kvikio.defaults.max_attempts() == 5
        kvikio.defaults.max_attempts_reset(4)
        assert kvikio.defaults.max_attempts() == 4
    assert before == kvikio.defaults.max_attempts()

    with pytest.raises(ValueError, match="positive integer"):
        kvikio.defaults.max_attempts_reset(0)
    with pytest.raises(OverflowError, match="negative value"):
        kvikio.defaults.max_attempts_reset(-1)


def test_http_status_codes():
    before = kvikio.defaults.http_status_codes()

    with kvikio.defaults.set_http_status_codes([500]):
        assert kvikio.defaults.http_status_codes() == [500]
        kvikio.defaults.http_status_codes_reset([429, 500])
        assert kvikio.defaults.http_status_codes() == [429, 500]
    assert before == kvikio.defaults.http_status_codes()

    with pytest.raises(TypeError):
        kvikio.defaults.http_status_codes_reset(0)

    with pytest.raises(TypeError):
        kvikio.defaults.http_status_codes_reset(["a"])
