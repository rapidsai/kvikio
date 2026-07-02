# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from kvikio.tools import remote_io_monitor as rim


class FakeSource(rim.NicCounterSource):
    """Counter source whose RX/TX grow by a fixed step on every read."""

    def __init__(self, names=("eth0", "eth1"), step=1 << 20):
        self._names = list(names)
        self._step = step
        self._calls = 0

    def read_counters(self):
        self._calls += 1
        base = self._calls * self._step
        return {n: rim.NicCounters(base, base) for n in self._names}


def test_source_kind_parse():
    assert rim.NicCounterSourceKind.parse("auto") is rim.NicCounterSourceKind.AUTO
    assert rim.NicCounterSourceKind.parse("SYSFS") is rim.NicCounterSourceKind.SYSFS
    assert rim.NicCounterSourceKind.parse("PsUtil") is rim.NicCounterSourceKind.PSUTIL


def test_create_auto_returns_a_source():
    source = rim.NicCounterSource.create()
    assert isinstance(source, rim.NicCounterSource)
    counters = source.read_counters()
    assert isinstance(counters, dict)
    for name, c in counters.items():
        assert isinstance(name, str)
        assert isinstance(c, rim.NicCounters)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="sysfs is Linux only"
)
def test_sysfs_source_reports_loopback():
    counters = rim.SysfsCounterSource().read_counters()
    assert "lo" in counters


def test_select_interfaces_respects_request():
    chosen = rim._select_interfaces(["lo", "eth0", "eth1"], ["eth1"])
    assert chosen == ["eth1"]


def test_csv_logger_sample_math(tmp_path):
    logger = rim.CsvBandwidthLogger(
        FakeSource(),
        str(tmp_path / "bw.csv"),
        freq_hz=50.0,
        interfaces=["eth0", "eth1"],
    )
    logger.open()
    sample = logger.sample()
    assert set(sample.rates_mibps) == {"eth0", "eth1"}
    assert all(rate > 0.0 for rate in sample.rates_mibps.values())
    assert sample.total_mibps == pytest.approx(sum(sample.rates_mibps.values()))
    logger.close()


def test_csv_logger_writes_csv(tmp_path):
    csv_path = tmp_path / "bw.csv"
    logger = rim.CsvBandwidthLogger(
        FakeSource(), str(csv_path), freq_hz=200.0, interfaces=["eth0", "eth1"]
    )
    logger.open()
    counter = {"n": 0}

    def stop():
        counter["n"] += 1
        return counter["n"] > 3

    logger.run_until(stop)
    logger.close()

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "monotonic_s,total_mibps,eth0,eth1"
    assert len(lines) >= 2  # header + at least one row


def test_csv_logger_rejects_bad_freq(tmp_path):
    with pytest.raises(ValueError):
        rim.CsvBandwidthLogger(FakeSource(), str(tmp_path / "x.csv"), freq_hz=0.0)


def test_cli_runs_app_and_writes_csv(tmp_path):
    csv_path = tmp_path / "cli.csv"
    rc = rim.main(
        [
            "--no-nvtx",
            "--csv",
            str(csv_path),
            "--freq-hz",
            "100",
            "--counter-source",
            "auto",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.15)",
        ]
    )
    assert rc == 0
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("monotonic_s,total_mibps,")


def test_cli_requires_app_command():
    assert rim.main(["--no-nvtx", "--csv", "x.csv"]) == 1


def test_cli_no_output_is_error():
    assert rim.main(["--no-nvtx", "--", sys.executable, "-c", "pass"]) == 1


def test_native_engine_if_built():
    nm = pytest.importorskip("kvikio._lib.nic_monitor")
    import time

    monitor = nm.NicBandwidthMonitor(100.0, ["lo"])
    assert not monitor.running()
    monitor.start()
    assert monitor.running()
    time.sleep(0.1)
    monitor.stop()
    assert not monitor.running()
    assert monitor.interfaces() == ["lo"]
