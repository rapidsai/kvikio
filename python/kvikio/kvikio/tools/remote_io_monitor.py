# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIC bandwidth monitor for KvikIO remote I/O.

Samples per-interface network byte counters while an application runs and
reports the receive rate in MiB/s. There are two independent outputs:

* **NVTX counter track** (default): emitted by the native C++ engine
  (:class:`kvikio._lib.nic_monitor.NicBandwidthMonitor`) so the bandwidth curve
  renders on the Nsight Systems timeline next to CUDA and KvikIO activity. The
  C++ sampling thread never holds the Python GIL.
* **CSV time series** (``--csv``): a dependency-free, pure-Python sampler that
  needs neither nsys nor the native engine, useful for a quick log and for
  testing.

Run it as a launcher that wraps any command (the application goes after ``--``)::

    nsys profile --trace=cuda,nvtx,osrt --trace-fork-before-exec=true \\
        --force-overwrite=true --output=netio \\
        python -m kvikio.tools.remote_io_monitor --freq-hz 50 -- <app> <args>

For embedded use inside your own Python process, the native engine is a context
manager: ``with kvikio._lib.nic_monitor.NicBandwidthMonitor(20.0): run_query()``
(GIL-immune, and the NVTX track lands in the app's own rows).
"""

from __future__ import annotations

import abc
import argparse
import enum
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Final, NamedTuple, Optional

_SYSFS_NET: Final[str] = "/sys/class/net"
_MIB: Final[float] = float(1 << 20)


def _read_text(path: str) -> str:
    """Read and return the full contents of a small text file."""
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Counter sources (used by the pure-Python CSV path; the native NVTX engine
# reads /sys directly in C++).
# ---------------------------------------------------------------------------


class NicCounters(NamedTuple):
    """Cumulative byte counters for a single network interface.

    Attributes
    ----------
    rx_bytes : int
        Total bytes received since boot.
    tx_bytes : int
        Total bytes transmitted since boot.
    """

    rx_bytes: int
    tx_bytes: int


class NicCounterSourceKind(enum.Enum):
    """Selects the implementation that reads NIC byte counters.

    Attributes
    ----------
    AUTO : int
        Use ``SYSFS`` when ``/sys/class/net`` is available, otherwise ``PSUTIL``.
    SYSFS : int
        Read ``/sys/class/net/<iface>/statistics/{rx,tx}_bytes`` directly.
    PSUTIL : int
        Read counters via ``psutil.net_io_counters(pernic=True)``.
    """

    AUTO = 0
    SYSFS = 1
    PSUTIL = 2

    @staticmethod
    def parse(name: str) -> NicCounterSourceKind:
        """Return the kind matching a case-insensitive name."""
        return NicCounterSourceKind[name.strip().upper()]


class NicCounterSource(abc.ABC):
    """Reads cumulative NIC byte counters, one entry per interface."""

    @abc.abstractmethod
    def read_counters(self) -> dict[str, NicCounters]:
        """Return current cumulative counters keyed by interface name."""

    @staticmethod
    def create(
        kind: NicCounterSourceKind = NicCounterSourceKind.AUTO,
    ) -> NicCounterSource:
        """Create a counter source.

        Parameters
        ----------
        kind
            Which implementation to use. ``AUTO`` prefers the sysfs reader when
            ``/sys/class/net`` exists and falls back to psutil.

        Returns
        -------
        NicCounterSource
            A ready-to-use source instance.

        Raises
        ------
        ValueError
            If ``kind`` is not a known source kind.
        """
        if kind is NicCounterSourceKind.AUTO:
            kind = (
                NicCounterSourceKind.SYSFS
                if os.path.isdir(_SYSFS_NET)
                else NicCounterSourceKind.PSUTIL
            )
        if kind is NicCounterSourceKind.SYSFS:
            return SysfsCounterSource()
        if kind is NicCounterSourceKind.PSUTIL:
            return PsutilCounterSource()
        raise ValueError(f"unknown counter source kind: {kind!r}")


class SysfsCounterSource(NicCounterSource):
    """Counter source backed by ``/sys/class/net/<iface>/statistics``."""

    def read_counters(self) -> dict[str, NicCounters]:
        """Read RX/TX byte counters for every interface under sysfs."""
        result: dict[str, NicCounters] = {}
        try:
            names = os.listdir(_SYSFS_NET)
        except OSError:
            return result
        for name in names:
            stats = os.path.join(_SYSFS_NET, name, "statistics")
            try:
                rx = int(_read_text(os.path.join(stats, "rx_bytes")))
                tx = int(_read_text(os.path.join(stats, "tx_bytes")))
            except (OSError, ValueError):
                # Interface vanished or exposes no byte counters; skip it.
                continue
            result[name] = NicCounters(rx, tx)
        return result


class PsutilCounterSource(NicCounterSource):
    """Counter source backed by ``psutil.net_io_counters(pernic=True)``.

    ``psutil`` is imported lazily, so it is only required when this source is
    used (the default ``sysfs``/``auto`` path has no such dependency).
    """

    def __init__(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "PsutilCounterSource requires psutil; install it with "
                "'pip install psutil', or use the default sysfs source."
            ) from exc
        self._psutil = psutil

    def read_counters(self) -> dict[str, NicCounters]:
        """Read RX/TX byte counters for every interface psutil reports."""
        per_nic = self._psutil.net_io_counters(pernic=True)
        return {
            name: NicCounters(stat.bytes_recv, stat.bytes_sent)
            for name, stat in per_nic.items()
        }


# ---------------------------------------------------------------------------
# CSV path: a pure-Python sampler (no native engine, no nsys).
# ---------------------------------------------------------------------------


class BandwidthSample(NamedTuple):
    """One bandwidth reading across the monitored interfaces.

    Attributes
    ----------
    monotonic_s : float
        ``time.monotonic()`` timestamp when the reading was taken.
    rates_mibps : dict[str, float]
        Receive rate in MiB/s for each monitored interface.
    total_mibps : float
        Sum of ``rates_mibps`` across interfaces, kept separate so that
        ``"total"`` is never mistaken for a real interface name.
    """

    monotonic_s: float
    rates_mibps: dict[str, float]
    total_mibps: float


class Sink(abc.ABC):
    """Destination for :class:`BandwidthSample` records."""

    @abc.abstractmethod
    def open(self, interfaces: Sequence[str]) -> None:
        """Prepare the sink for the given fixed set of interfaces."""

    @abc.abstractmethod
    def emit(self, sample: BandwidthSample) -> None:
        """Write one sample to the destination."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the sink."""


class CsvSink(Sink):
    """Appends samples to a CSV file, one column per interface plus total."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = None
        self._interfaces: list[str] = []

    def open(self, interfaces: Sequence[str]) -> None:
        """Open the file and write the header row."""
        self._interfaces = list(interfaces)
        self._file = open(self._path, "w", buffering=1)
        header = ["monotonic_s", "total_mibps", *self._interfaces]
        self._file.write(",".join(header) + "\n")

    def emit(self, sample: BandwidthSample) -> None:
        """Append one row for ``sample``."""
        if self._file is None:
            return
        row = [f"{sample.monotonic_s:.6f}", f"{sample.total_mibps:.3f}"]
        row += [f"{sample.rates_mibps.get(n, 0.0):.3f}" for n in self._interfaces]
        self._file.write(",".join(row) + "\n")

    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None


def _iface_is_up(name: str) -> bool:
    """Return whether an interface is administratively up.

    Reads ``/sys/class/net/<name>/operstate``. When the state cannot be read the
    interface is kept, since an unknown state is not a reason to drop it.
    """
    try:
        state = _read_text(os.path.join(_SYSFS_NET, name, "operstate")).strip()
    except OSError:
        return True
    return state != "down"


def _select_interfaces(
    available: Iterable[str], requested: Optional[Sequence[str]]
) -> list[str]:
    """Choose which interfaces to monitor.

    When ``requested`` is given it is used verbatim. Otherwise all UP
    non-loopback interfaces are selected; traffic is never used as a filter
    because an idle-at-start interface can become active later.
    """
    if requested:
        return list(requested)
    return sorted(n for n in available if n != "lo" and _iface_is_up(n))


class CsvBandwidthLogger:
    """Pure-Python sampler that writes per-interface MiB/s to a CSV file.

    This path needs neither the native engine nor nsys, so it is testable in
    pure Python and works as a dependency-free fallback.

    Parameters
    ----------
    source
        Counter source to read.
    path
        CSV file to write.
    freq_hz
        Sampling frequency in hertz.
    interfaces
        Interfaces to log. ``None`` selects all UP non-loopback interfaces at
        :meth:`open` time.
    """

    def __init__(
        self,
        source: NicCounterSource,
        path: str,
        *,
        freq_hz: float = 20.0,
        interfaces: Optional[Sequence[str]] = None,
    ) -> None:
        if freq_hz <= 0.0:
            raise ValueError("freq_hz must be positive")
        self._source = source
        self._sink = CsvSink(path)
        self._freq_hz = float(freq_hz)
        self._requested = list(interfaces) if interfaces is not None else None
        self._interfaces: list[str] = []
        self._prev: dict[str, NicCounters] = {}
        self._prev_t = 0.0
        self._opened = False

    def open(self) -> None:
        """Select interfaces, take a baseline reading, and open the CSV file."""
        if self._opened:
            return
        counters = self._source.read_counters()
        self._interfaces = _select_interfaces(counters.keys(), self._requested)
        self._prev = counters
        self._prev_t = time.monotonic()
        self._sink.open(self._interfaces)
        self._opened = True

    def close(self) -> None:
        """Close the CSV file. Safe to call more than once."""
        if not self._opened:
            return
        self._sink.close()
        self._opened = False

    def sample(self) -> BandwidthSample:
        """Read the source and return the rate since the previous sample."""
        now = time.monotonic()
        counters = self._source.read_counters()
        dt = now - self._prev_t
        rates: dict[str, float] = {}
        for name in self._interfaces:
            cur = counters.get(name)
            prev = self._prev.get(name)
            if cur is None or prev is None or dt <= 0.0:
                rates[name] = 0.0
            else:
                delta = max(0, cur.rx_bytes - prev.rx_bytes)
                rates[name] = delta / dt / _MIB
        self._prev = counters
        self._prev_t = now
        return BandwidthSample(now, rates, sum(rates.values()))

    def run_until(self, stop: Callable[[], bool]) -> None:
        """Sample and write at ``freq_hz`` until ``stop()`` is true.

        Writes one final sample after the loop to capture the tail/drain.
        """
        if not self._opened:
            raise RuntimeError("CsvBandwidthLogger.run_until requires open()")
        interval = 1.0 / self._freq_hz
        next_t = time.monotonic()
        while not stop():
            self._sink.emit(self.sample())
            next_t += interval
            time.sleep(max(0.0, next_t - time.monotonic()))
        self._sink.emit(self.sample())


# ---------------------------------------------------------------------------
# NVTX path: the native C++ engine.
# ---------------------------------------------------------------------------


def _make_native_monitor(freq_hz: float, interfaces: Optional[Sequence[str]]):
    """Construct the native NVTX engine, with a clear error if unavailable."""
    try:
        from kvikio._lib.nic_monitor import NicBandwidthMonitor
    except ImportError as exc:
        raise RuntimeError(
            "The native NIC monitor is unavailable; build KvikIO "
            "(build-all) to enable the NVTX counter track, or run with "
            "--no-nvtx --csv for a pure-Python CSV log."
        ) from exc
    return NicBandwidthMonitor(freq_hz, list(interfaces) if interfaces else None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (monitor flags only; app goes after --)."""
    parser = argparse.ArgumentParser(
        prog="python -m kvikio.tools.remote_io_monitor",
        description=(
            "Sample NIC bandwidth while running an application. By default it "
            "emits an NVTX counter track (native engine); add --csv for a "
            "pure-Python CSV log. Put the application command after '--'."
        ),
    )
    parser.add_argument(
        "--iface",
        action="append",
        metavar="NAME",
        help="Interface to monitor; repeatable. Default: all UP non-loopback.",
    )
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=20.0,
        help="Sampling frequency in hertz (default: %(default)s).",
    )
    parser.add_argument(
        "--no-nvtx",
        action="store_true",
        help="Do not emit the NVTX counter track (use with --csv).",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Also write samples to this CSV file (pure-Python, no nsys needed).",
    )
    parser.add_argument(
        "--counter-source",
        default="auto",
        choices=["auto", "sysfs", "psutil"],
        help="Counter source for the CSV path (default: %(default)s).",
    )
    return parser


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at the first ``--`` into (monitor args, app argv)."""
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run an application under the NIC monitor (CLI entry point).

    Parameters
    ----------
    argv
        Argument list excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        The application's exit code, or 1 on a usage error.
    """
    monitor_argv, app_argv = _split_argv(
        sys.argv[1:] if argv is None else argv
    )
    args = _build_arg_parser().parse_args(monitor_argv)
    if not app_argv:
        print("error: no application command given after '--'", file=sys.stderr)
        return 1

    nvtx_enabled = not args.no_nvtx
    csv_enabled = args.csv is not None
    if not nvtx_enabled and not csv_enabled:
        print(
            "error: --no-nvtx given without --csv leaves nothing to do",
            file=sys.stderr,
        )
        return 1

    # Build outputs before launching the app so a failure (e.g. native engine
    # not built) does not leave an orphaned child process.
    monitor = _make_native_monitor(args.freq_hz, args.iface) if nvtx_enabled else None
    csv_logger = None
    if csv_enabled:
        source = NicCounterSource.create(
            NicCounterSourceKind.parse(args.counter_source)
        )
        csv_logger = CsvBandwidthLogger(
            source, args.csv, freq_hz=args.freq_hz, interfaces=args.iface
        )
        csv_logger.open()

    if monitor is not None:
        monitor.start()
    proc = subprocess.Popen(app_argv)

    def _forward(signum, frame) -> None:
        # Pass the signal to the child so it can shut down; the loop ends when
        # the child exits.
        proc.send_signal(signum)

    old_int = signal.signal(signal.SIGINT, _forward)
    old_term = signal.signal(signal.SIGTERM, _forward)
    try:
        if csv_logger is not None:
            csv_logger.run_until(lambda: proc.poll() is not None)
        else:
            interval = min(0.05, 1.0 / args.freq_hz)
            while proc.poll() is None:
                time.sleep(interval)
    finally:
        if monitor is not None:
            monitor.stop()
        if csv_logger is not None:
            csv_logger.close()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    return proc.returncode if proc.returncode is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
