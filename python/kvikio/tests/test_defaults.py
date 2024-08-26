# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

import kvikio.defaults


@pytest.mark.skipif(
    kvikio.defaults.compat_mode(),
    reason="cannot test `compat_mode` when already running in compatibility mode",
)
def test_compat_mode():
    """Test changing `compat_mode`"""

    before = kvikio.defaults.compat_mode()
    with kvikio.defaults.set_compat_mode(True):
        assert kvikio.defaults.compat_mode()
        kvikio.defaults.compat_mode_reset(False)
        assert not kvikio.defaults.compat_mode()
    assert before == kvikio.defaults.compat_mode()


def test_num_threads():
    """Test changing `num_threads`"""

    before = kvikio.defaults.get_num_threads()
    with kvikio.defaults.set_num_threads(3):
        assert kvikio.defaults.get_num_threads() == 3
        kvikio.defaults.num_threads_reset(4)
        assert kvikio.defaults.get_num_threads() == 4
    assert before == kvikio.defaults.get_num_threads()


def test_task_size():
    """Test changing `task_size`"""

    before = kvikio.defaults.task_size()
    with kvikio.defaults.set_task_size(3):
        assert kvikio.defaults.task_size() == 3
        kvikio.defaults.task_size_reset(4)
        assert kvikio.defaults.task_size() == 4
    assert before == kvikio.defaults.task_size()


def test_gds_threshold():
    """Test changing `gds_threshold`"""

    before = kvikio.defaults.gds_threshold()
    with kvikio.defaults.set_gds_threshold(3):
        assert kvikio.defaults.gds_threshold() == 3
        kvikio.defaults.gds_threshold_reset(4)
        assert kvikio.defaults.gds_threshold() == 4
    assert before == kvikio.defaults.gds_threshold()
