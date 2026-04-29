# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-flight batch-metrics snapshotter.

The orchestrator runs on the head node and shares a filesystem with the
worker logs (``outputs/<jobid>/logs/*prefill_w*.out`` etc.), so we can
poll those logs from a background Python thread without ssh / scp /
container hops. Every ``interval_seconds`` we incrementally re-parse the
freshly appended bytes (see ``LogState`` in :mod:`batch_metrics`) and
overwrite ``batch_metrics.png`` in place.

The snapshotter is started by :class:`BenchmarkStageMixin` right before
the benchmark srun is launched, and stopped right after it exits. It is
fully best-effort: any exception from parsing or rendering is logged and
swallowed, never propagated up to fail the benchmark.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from srtctl.analysis.batch_metrics import LogState, process_single_run

logger = logging.getLogger(__name__)


class LiveMetricsSnapshotter:
    """Background thread that periodically refreshes ``batch_metrics.png``.

    Usage::

        snap = LiveMetricsSnapshotter(log_dir, interval_seconds=60)
        snap.start(stop_event)
        try:
            ...  # run benchmark
        finally:
            snap.stop()

    The snapshotter exits whenever ``stop_event`` is set, the thread is
    explicitly ``stop()``-ed, or the wait timeout elapses with the parent
    benchmark already finished.
    """

    def __init__(
        self,
        log_dir: Path,
        interval_seconds: int = 60,
        downsample: int = 1,
        output_filename: str = "batch_metrics.png",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.interval_seconds = max(5, int(interval_seconds))
        self.downsample = max(1, int(downsample))
        self.output_path = self.log_dir / output_filename
        self._state = LogState(log_dir=self.log_dir)
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._tick_count = 0

    def start(self, stop_event: threading.Event | None = None) -> None:
        """Start the background snapshotter thread.

        If ``stop_event`` is provided, the thread will also exit when
        that event is set (typically the orchestrator's global stop
        signal).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = stop_event or threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="LiveMetricsSnapshotter",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Live batch-metrics snapshotter started: log_dir=%s interval=%ds output=%s",
            self.log_dir,
            self.interval_seconds,
            self.output_path,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the snapshotter to exit and wait for it to drain."""
        if self._stop_event is not None:
            self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Live metrics snapshotter did not exit within %.1fs", timeout)
        self._thread = None

    def _tick(self) -> None:
        """Run a single snapshot pass. Errors are logged but never raised."""
        try:
            ok = process_single_run(
                log_dir=self.log_dir,
                output_path=self.output_path,
                downsample=self.downsample,
                state=self._state,
            )
            self._tick_count += 1
            if ok:
                logger.debug(
                    "Live metrics tick #%d wrote %s (prefill_files=%d, decode_files=%d)",
                    self._tick_count,
                    self.output_path,
                    len(self._state.prefill_files),
                    len(self._state.decode_files),
                )
            else:
                logger.debug(
                    "Live metrics tick #%d: no batch lines parsed yet (workers still warming up?)",
                    self._tick_count,
                )
        except Exception as e:  # never let snapshot failures affect the benchmark
            logger.warning("Live metrics snapshot failed: %s", e, exc_info=False)

    def _run(self) -> None:
        assert self._stop_event is not None
        # Don't immediately tick — give workers a few seconds to write
        # their first batch lines so the very first PNG isn't empty.
        first_delay = min(self.interval_seconds, 15)
        if self._stop_event.wait(timeout=first_delay):
            return
        while not self._stop_event.is_set():
            self._tick()
            if self._stop_event.wait(timeout=self.interval_seconds):
                break
        # Final tick on the way out so the PNG reflects the very end of
        # the run (post-benchmark / post-cleanup).
        self._tick()
        logger.info("Live metrics snapshotter exited after %d ticks", self._tick_count)
