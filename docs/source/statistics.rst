Statistics
==========

.. currentmodule:: kvikio.statistics

KvikIO can report what a run did: how many operations, how many bytes, how long it was
busy, and which backend carried the work.

A monitor accumulates for as long as it exists, and :meth:`~SummaryMonitor.get`
reads the totals so far.

.. literalinclude:: ../../python/kvikio/examples/hello_world.py
    :language: python
    :start-at: import cupy

Printing a summary, or calling :meth:`~Summary.report`, gives a report meant to be read
by a person:

.. code-block:: text

    KvikIO I/O summary (LOGICAL)
      wall time            238.22 ms
      busy time            5.66 ms (2.38 % of the wall time)
      busy bandwidth       565.50 kB/s
      operations           5 (4 read, 1 write)
      mean duration        1.13 ms
      bytes                3.12 KiB of 3.12 KiB requested (2.34 KiB read, 800 B written)
      errors               0
      backend POSIX        3.12 KiB in 5 ops, 5.66 ms, 565.50 kB/s
      backend GDS          unused
      backend MMAP         unused
      backend REMOTE_HTTP  unused
      backend REMOTE_HDFS  unused

Busy time and bandwidth
-----------------------

**Busy time** is the union of the operations' spans, so overlapping work counts once and
the gaps between calls count as idle.

**Busy bandwidth** divides the bytes by that rather than by the wall time. A program that
reads for 10 ms and then computes for 90 ms is doing I/O at its storage's speed for a
tenth of its life, and dividing by the wall time would report it as ten times slower than
it is. Multiply by :attr:`~Summary.busy_fraction` to recover the whole-span rate.

Calls or transfers
------------------

A monitor counts one operation per user-facing call by default. A call that KvikIO splits
across its thread pool is one row, however many reads it issued underneath, and its span
runs from submission to completion. Under load most of that span is the wait for a worker
rather than the transfer.

Passing :attr:`ObservationKind.PHYSICAL` counts one operation per transfer instead: one
thread-pool task locally, one HTTP range request remotely. Its span starts when a worker
picks the task up, or when the request goes on the wire.

.. code-block:: python

    calls = kvikio.SummaryMonitor(kvikio.ObservationKind.LOGICAL)
    transfers = kvikio.SummaryMonitor(kvikio.ObservationKind.PHYSICAL)

:attr:`Summary.kind` says which a summary is over, and the report leads with it, so two
summaries in a log are never confused for one another.

Both see the same bytes and the same errors. What differs is the count and the durations,
so eight concurrent reads over a two-thread pool look like this:

.. code-block:: text

                             calls      transfers
    num_ops                      8             64
    busy (ms)               14.657         14.581
    total_duration (ms)     79.383         28.941

``busy``, the union of the spans, is the same because the calls and the transfers are in
flight over the same stretch of wall clock. ``total_duration``, the sum of the spans, is
not. The 50 ms between the two is what the calls spent queueing.

Which to pick follows from the question. Use the calls for how many I/Os a program issued
and how long each took as it experienced them. Use the transfers for how well the device
or the link was kept busy, and for bandwidth over time, since a call's bytes would
otherwise be charged to the moment it returned rather than to the moments they moved.

Two things to know. A physical monitor pays its per-observation cost once per task rather
than once per call, so watching a run of large split reads costs proportionally more.
And the two remote backends describe a retried request differently. ``MULTI_POLL``
reports one transfer per attempt, so the backoff between them belongs to neither.
``EASY_THREADPOOL`` retries inside the call that the transfer is measured around, so the
attempts and the waits between them are one transfer, and that transfer's duration
includes time when nothing was on the wire.

Getting a summary out of the process
------------------------------------

:meth:`~Summary.to_json` gives the same content for anything that would rather
parse it, with the timestamps against the wall clock so another program can line the
summary up with its own log.

:meth:`~Summary.serialize` and :meth:`~Summary.deserialize` move a summary
between processes. They are exact, so one that has been through a pipe is still a valid
``previous`` for :meth:`~Summary.since`, and they are what pickling uses.

.. code-block:: python

    raw = summary.serialize()
    assert kvikio.Summary.deserialize(raw) == summary

The bytes are not a wire format. They are a copy of the C++ struct, readable only by the
same architecture and the same build of KvikIO, and anything else is refused rather than
misread.

Summaries are not additive. Most fields could be added across processes, but busy time
cannot, since two processes are genuinely busy at the same moment.
