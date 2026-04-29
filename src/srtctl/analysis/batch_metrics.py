# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse SGLang prefill/decode worker logs and render time-series plots.

This module exposes a small library API used by both:

- ``plot_batch_metrics.py`` (repo-root CLI, post-mortem)
- ``srtctl.analysis.live_metrics`` (in-flight live snapshotter)

The default plot view is **cluster-wide aggregation**: prefill metrics are
summed across worker leaders, decode DP-rank metrics are scaled up by the
DP factor inferred from the log volume. This produces a single line per
metric that reflects the whole serving cluster instead of N noisy
per-worker traces.

The parsers also support **incremental** reads: callers can pass a
``LogState`` to remember per-file byte offsets and accumulated rows, so
re-parsing a long benchmark every minute stays cheap.

The line format we parse looks like (one of two timestamp variants)::

    p0\\x1b[2m2026-04-27T23:03:15.250907Z\\x1b[0m ... Prefill batch,
        #new-seq: 1, #new-token: 256, ..., #running-req: 0,
        #queue-req: 0, #prealloc-req: 0, #inflight-req: 0,
        input throughput (token/s): 0.00,

    [2025-11-04 05:31:43 DP0 TP0 EP0] Decode batch, #running-req: 1,
        #full token: 7424, full token usage: 0.00,
        ..., gen throughput (token/s): 0.03, #queue-req: 0,
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# ============================================================================
# Field definitions
# ============================================================================

PREFILL_METRICS: dict[str, str] = {
    "#new-seq": r"#new-seq:\s*([\d.]+)",
    "#new-token": r"#new-token:\s*([\d.]+)",
    "#cached-token": r"#cached-token:\s*([\d.]+)",
    "full token usage": r"full token usage:\s*([\d.]+)",
    "mamba usage": r"mamba usage:\s*([\d.]+)",
    "#running-req": r"#running-req:\s*([\d.]+)",
    "#queue-req": r"#queue-req:\s*([\d.]+)",
    "#prealloc-req": r"#prealloc-req:\s*([\d.]+)",
    "#inflight-req": r"#inflight-req:\s*([\d.]+)",
    "input throughput (token/s)": r"input throughput \(token/s\):\s*([\d.]+)",
}

DECODE_METRICS: dict[str, str] = {
    "#running-req": r"#running-req:\s*([\d.]+)",
    "#full token": r"#full token:\s*([\d.]+)",
    "full token usage": r"full token usage:\s*([\d.]+)",
    "mamba num": r"mamba num:\s*([\d.]+)",
    "mamba usage": r"mamba usage:\s*([\d.]+)",
    "pre-allocated usage": r"pre-allocated usage:\s*([\d.]+)",
    "#prealloc-req": r"#prealloc-req:\s*([\d.]+)",
    "#transfer-req": r"#transfer-req:\s*([\d.]+)",
    "#retracted-req": r"#retracted-req:\s*([\d.]+)",
    "gen throughput (token/s)": r"gen throughput \(token/s\):\s*([\d.]+)",
    "#queue-req": r"#queue-req:\s*([\d.]+)",
}

PREFILL_KEYWORD = "Prefill batch"
DECODE_KEYWORD = "Decode batch"

# ANSI-escaped ISO timestamp: \x1b[2m2026-02-26T08:42:59.339496Z\x1b[0m
_TS_ANSI = re.compile(r"\[2m(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}\.\d+)")
# Plain bracket timestamp: [2025-11-04 05:31:43 DP0 TP0 EP0]
_TS_BRACKET = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# Compiled regex per metric key, lazily.
_COMPILED_PATTERNS: dict[str, re.Pattern[str]] = {}


def _pattern(key: str, raw: str) -> re.Pattern[str]:
    cached = _COMPILED_PATTERNS.get(key)
    if cached is None:
        cached = re.compile(raw)
        _COMPILED_PATTERNS[key] = cached
    return cached


def _parse_timestamp(line: str) -> datetime | None:
    m = _TS_ANSI.search(line)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None
    m = _TS_BRACKET.search(line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def _extract(line: str, key: str, raw_pattern: str) -> float | None:
    m = _pattern(key, raw_pattern).search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ============================================================================
# Per-file parser with incremental support
# ============================================================================


@dataclass
class FileSeries:
    """Time-series accumulated from a single worker log file.

    Mutated in place by ``parse_file`` so the snapshotter can keep one
    ``FileSeries`` per file across ticks and only read freshly appended
    bytes each time.
    """

    path: Path
    timestamps: list[datetime] = field(default_factory=list)
    metrics: dict[str, list[float | None]] = field(default_factory=dict)
    byte_offset: int = 0

    def initialise_metrics(self, names: Iterable[str]) -> None:
        for n in names:
            self.metrics.setdefault(n, [])

    def empty(self) -> bool:
        return not self.timestamps


def parse_file(series: FileSeries, keyword: str, metrics_def: dict[str, str]) -> int:
    """Append rows parsed from new bytes in ``series.path`` to ``series``.

    Returns:
        Number of new rows appended.
    """
    series.initialise_metrics(metrics_def.keys())
    path = series.path
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    if size <= series.byte_offset:
        return 0

    new_rows = 0
    try:
        with open(path, errors="replace") as f:
            f.seek(series.byte_offset)
            for line in f:
                if keyword not in line:
                    continue
                ts = _parse_timestamp(line)
                if ts is None:
                    continue

                values: dict[str, float | None] = {}
                for name, raw in metrics_def.items():
                    values[name] = _extract(line, name, raw)
                if all(v is None for v in values.values()):
                    continue

                series.timestamps.append(ts)
                for name in metrics_def:
                    series.metrics[name].append(values[name])
                new_rows += 1

            series.byte_offset = f.tell()
    except OSError as e:
        logger.warning("failed to read %s: %s", path, e)
        return 0

    return new_rows


# ============================================================================
# Log directory discovery
# ============================================================================


def find_log_files(log_dir: str | Path) -> tuple[list[Path], list[Path]]:
    """Find prefill / decode worker log files in ``log_dir``.

    Mirrors the heuristic of ``plot_batch_metrics.py``: filenames matching
    ``*prefill*`` go to the prefill list, ``*decode*`` to the decode list,
    and ``*_agg_*`` files (aggregated mode) are duplicated into both since
    they carry both prefill and decode batch lines.
    """
    log_dir = Path(log_dir)
    prefill: list[Path] = []
    decode: list[Path] = []
    agg: list[Path] = []

    if not log_dir.is_dir():
        return prefill, decode

    for entry in sorted(os.listdir(log_dir)):
        if not (entry.endswith(".out") or entry.endswith(".err")):
            continue
        path = log_dir / entry
        if "prefill" in entry:
            prefill.append(path)
        elif "decode" in entry:
            decode.append(path)
        elif "_agg_" in entry:
            agg.append(path)

    for path in agg:
        prefill.append(path)
        decode.append(path)

    return prefill, decode


# ============================================================================
# State for incremental snapshotting
# ============================================================================


@dataclass
class LogState:
    """Per-log-dir incremental state shared across snapshot ticks."""

    log_dir: Path
    prefill_files: dict[Path, FileSeries] = field(default_factory=dict)
    decode_files: dict[Path, FileSeries] = field(default_factory=dict)

    def refresh(self) -> tuple[int, int]:
        """Re-discover files and parse only the newly appended bytes.

        Returns:
            (new_prefill_rows, new_decode_rows) appended this call.
        """
        pf_paths, dc_paths = find_log_files(self.log_dir)

        new_pf = 0
        for p in pf_paths:
            series = self.prefill_files.setdefault(p, FileSeries(path=p))
            new_pf += parse_file(series, PREFILL_KEYWORD, PREFILL_METRICS)

        new_dc = 0
        for p in dc_paths:
            series = self.decode_files.setdefault(p, FileSeries(path=p))
            new_dc += parse_file(series, DECODE_KEYWORD, DECODE_METRICS)

        return new_pf, new_dc

    def has_data(self) -> bool:
        return any(not s.empty() for s in self.prefill_files.values()) or any(
            not s.empty() for s in self.decode_files.values()
        )


# ============================================================================
# Cluster-wide aggregation
# ============================================================================


def _bucket_seconds(ts: datetime, bucket_s: int) -> datetime:
    """Floor ``ts`` to a ``bucket_s``-second bucket boundary."""
    epoch = ts.timestamp()
    floored = (int(epoch) // bucket_s) * bucket_s
    return datetime.fromtimestamp(floored, tz=ts.tzinfo)


@dataclass
class AggregatedSeries:
    """Cluster-wide aggregated time-series, ready for matplotlib.

    Each entry in ``points`` holds one bucket: a list of (bucket_ts,
    aggregated_value) tuples per metric name.
    """

    bucket_seconds: int
    n_workers_seen: int
    points: dict[str, list[tuple[datetime, float]]] = field(default_factory=dict)

    def empty(self) -> bool:
        return all(len(v) == 0 for v in self.points.values())


def aggregate_prefill(
    files: Iterable[FileSeries],
    bucket_seconds: int = 5,
) -> AggregatedSeries:
    """Aggregate prefill metrics across worker files into a cluster-wide series.

    Strategy (mirrors ``current/dsv4/coreweave-352/timeseries/parse_and_plot.py``):

    1. For each (bucket, worker) pair, compute the **mean** of each metric
       across however many lines that worker logged in that bucket. This
       collapses bursty periods where one worker prints multiple lines per
       second.
    2. Then **sum** across workers in the same bucket. The result is a
       single per-cluster instantaneous value at each bucket boundary.

    For "rate" metrics (throughput) and "count" metrics (#running-req,
    #queue-req, #prealloc-req, #inflight-req) the sum-across-workers
    semantics is the right one. For "ratio" metrics (full token usage,
    mamba usage) we still expose a per-worker mean; we keep the same
    sum-then-fix-up implementation but emit a separate ``-mean`` series.
    """
    files = list(files)
    if not files:
        return AggregatedSeries(bucket_seconds=bucket_seconds, n_workers_seen=0)

    # bucket_ts -> worker_id -> metric -> list of (sum, count)
    by_bucket: dict[datetime, dict[Path, dict[str, list[float]]]] = {}

    sum_metrics = {
        "#new-seq",
        "#new-token",
        "#cached-token",
        "#running-req",
        "#queue-req",
        "#prealloc-req",
        "#inflight-req",
        "input throughput (token/s)",
    }
    mean_metrics = {"full token usage", "mamba usage"}

    workers_seen: set[Path] = set()
    for series in files:
        if series.empty():
            continue
        workers_seen.add(series.path)
        for i, ts in enumerate(series.timestamps):
            bucket = _bucket_seconds(ts, bucket_seconds)
            wmap = by_bucket.setdefault(bucket, {}).setdefault(series.path, {})
            for metric in PREFILL_METRICS:
                v = series.metrics[metric][i]
                if v is None:
                    continue
                wmap.setdefault(metric, []).append(v)

    out: AggregatedSeries = AggregatedSeries(
        bucket_seconds=bucket_seconds,
        n_workers_seen=len(workers_seen),
    )
    for metric in PREFILL_METRICS:
        out.points[metric] = []

    for bucket in sorted(by_bucket):
        per_worker = by_bucket[bucket]
        for metric in PREFILL_METRICS:
            worker_means: list[float] = []
            for vals in per_worker.values():
                samples = vals.get(metric)
                if not samples:
                    continue
                worker_means.append(sum(samples) / len(samples))
            if not worker_means:
                continue
            if metric in sum_metrics:
                value = sum(worker_means)
            elif metric in mean_metrics:
                value = sum(worker_means) / len(worker_means)
            else:
                value = sum(worker_means) / len(worker_means)
            out.points[metric].append((bucket, value))

    return out


def aggregate_decode(
    files: Iterable[FileSeries],
    bucket_seconds: int = 5,
) -> AggregatedSeries:
    """Aggregate decode metrics across DP ranks into a cluster-wide series.

    Decode worker logs do **not** carry an explicit DP-rank tag in the
    line prefix — all DP ranks within a worker dump to the same stdout.
    So a single bucket typically contains ``DP_factor * steps_per_bucket``
    log lines.

    Strategy:

    1. Group by ``(bucket, file)``: take the **mean** to recover the
       per-rank instantaneous value within the bucket.
    2. Multiply by an inferred DP factor (max lines per bucket per file
       within this snapshot, capped between 1 and 64) to scale up to
       worker-total ``running-req`` / ``gen throughput`` / ``#full token``.

    Ratio metrics (``full token usage``, ``mamba usage``,
    ``pre-allocated usage``) are reported as the per-rank mean (no DP
    scaling) because they are bounded in [0, 1].
    """
    files = list(files)
    if not files:
        return AggregatedSeries(bucket_seconds=bucket_seconds, n_workers_seen=0)

    # bucket -> file -> metric -> list of values
    by_bucket: dict[datetime, dict[Path, dict[str, list[float]]]] = {}
    # bucket -> file -> total line count (used for DP inference)
    line_counts: dict[datetime, dict[Path, int]] = {}

    workers_seen: set[Path] = set()
    for series in files:
        if series.empty():
            continue
        workers_seen.add(series.path)
        for i, ts in enumerate(series.timestamps):
            bucket = _bucket_seconds(ts, bucket_seconds)
            wmap = by_bucket.setdefault(bucket, {}).setdefault(series.path, {})
            line_counts.setdefault(bucket, {}).setdefault(series.path, 0)
            line_counts[bucket][series.path] += 1
            for metric in DECODE_METRICS:
                v = series.metrics[metric][i]
                if v is None:
                    continue
                wmap.setdefault(metric, []).append(v)

    # Infer DP factor: max #lines/bucket/file across the run, clipped 1..64.
    if line_counts:
        max_lines = max((c for buckets in line_counts.values() for c in buckets.values()), default=1)
    else:
        max_lines = 1
    # Within a 5-second bucket, decode steps fire roughly once per ~50 ms,
    # so per-rank we expect ~100 lines/bucket. We assume
    # DP factor ≈ max_lines / steps_per_bucket. We don't know
    # steps_per_bucket exactly, but in practice DP rank count is the
    # dominating multiplier for typical benchmarks. We clamp to 1..64 so
    # absurd inferences don't blow up the y-axis.
    inferred_dp = max(1, min(64, max_lines // 10 or 1))

    rate_metrics = {
        "#running-req",
        "#full token",
        "#prealloc-req",
        "#transfer-req",
        "#retracted-req",
        "#queue-req",
        "gen throughput (token/s)",
    }
    ratio_metrics = {"full token usage", "mamba usage", "pre-allocated usage", "mamba num"}

    out = AggregatedSeries(bucket_seconds=bucket_seconds, n_workers_seen=len(workers_seen))
    for metric in DECODE_METRICS:
        out.points[metric] = []

    for bucket in sorted(by_bucket):
        per_file = by_bucket[bucket]
        for metric in DECODE_METRICS:
            file_means: list[float] = []
            for vals in per_file.values():
                samples = vals.get(metric)
                if not samples:
                    continue
                file_means.append(sum(samples) / len(samples))
            if not file_means:
                continue
            per_rank_mean = sum(file_means) / len(file_means)
            if metric in rate_metrics:
                value = per_rank_mean * inferred_dp
            elif metric in ratio_metrics:
                value = per_rank_mean
            else:
                value = per_rank_mean
            out.points[metric].append((bucket, value))

    out.points["__inferred_dp__"] = [(datetime.fromtimestamp(0), float(inferred_dp))]
    return out


# ============================================================================
# Plotting
# ============================================================================


def _elapsed_seconds(stamps: list[datetime]) -> list[float]:
    if not stamps:
        return []
    t0 = stamps[0]
    return [(t - t0).total_seconds() for t in stamps]


def render_plot(
    prefill_agg: AggregatedSeries,
    decode_agg: AggregatedSeries,
    output_path: str | Path,
    title_prefix: str = "",
    downsample: int = 1,
) -> None:
    """Render the cluster-wide aggregated series as a 2-column PNG.

    Left column: prefill metrics (sum across worker leaders per bucket).
    Right column: decode metrics (per-rank mean × inferred DP factor).

    No data on a side renders a placeholder text panel so the layout
    stays consistent across snapshot ticks.
    """
    # Imported lazily so ``import srtctl.analysis.batch_metrics`` does not
    # require matplotlib in environments that only need the parsers.
    import matplotlib

    matplotlib.use("Agg")  # headless backend for SLURM/host
    import matplotlib.pyplot as plt

    prefill_metric_names = list(PREFILL_METRICS.keys())
    decode_metric_names = list(DECODE_METRICS.keys())
    n_rows = max(len(prefill_metric_names), len(decode_metric_names))

    fig, axes = plt.subplots(n_rows, 2, figsize=(22, 3.5 * n_rows), squeeze=False)
    title = f"{title_prefix}Cluster-wide Batch Metrics" if title_prefix else "Cluster-wide Batch Metrics"
    inferred_dp_pts = decode_agg.points.get("__inferred_dp__", [])
    inferred_dp = int(inferred_dp_pts[0][1]) if inferred_dp_pts else 1
    subtitle = (
        f"prefill: {prefill_agg.n_workers_seen} workers (sum) · "
        f"decode: {decode_agg.n_workers_seen} workers × DP≈{inferred_dp}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", y=1.0)

    def _draw(ax, series_pts: list[tuple[datetime, float]], label: str) -> bool:
        if downsample > 1:
            series_pts = series_pts[::downsample]
        if not series_pts:
            return False
        stamps = [p[0] for p in series_pts]
        values = [p[1] for p in series_pts]
        elapsed = _elapsed_seconds(stamps)
        ax.plot(elapsed, values, color="C0", linewidth=1.0, alpha=0.9, label=label)
        return True

    for row, metric in enumerate(prefill_metric_names):
        ax = axes[row][0]
        has = _draw(ax, prefill_agg.points.get(metric, []), label=metric)
        ax.set_title(f"Prefill: {metric}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Elapsed (s)", fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if not has:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")

    for row in range(len(prefill_metric_names), n_rows):
        axes[row][0].set_visible(False)

    for row, metric in enumerate(decode_metric_names):
        ax = axes[row][1]
        has = _draw(ax, decode_agg.points.get(metric, []), label=metric)
        title_label = f"Decode (×DP={inferred_dp}): {metric}" if metric != "full token usage" else f"Decode: {metric}"
        ax.set_title(title_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Elapsed (s)", fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if not has:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")

    for row in range(len(decode_metric_names), n_rows):
        axes[row][1].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish write: render to a sibling tmp file then rename, so
    # concurrent readers (e.g. an image viewer with auto-reload) never
    # see a half-written PNG. We append ``.tmp`` to the full filename
    # rather than using ``with_suffix`` (which replaces the extension and
    # confuses matplotlib's format inference).
    tmp = output_path.parent / (output_path.name + ".tmp")
    fig.savefig(tmp, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    os.replace(tmp, output_path)


# ============================================================================
# High-level entry points
# ============================================================================


def get_run_title(log_dir: str | Path) -> str:
    parts = Path(log_dir).resolve().parts
    for p in reversed(parts):
        if p != "logs":
            return f"{p} - "
    return ""


def process_single_run(
    log_dir: str | Path,
    output_path: str | Path | None = None,
    downsample: int = 1,
    state: LogState | None = None,
) -> bool:
    """Parse the log dir end-to-end and refresh the PNG. Returns ``True`` on success.

    When ``state`` is supplied, the parse is **incremental** — only newly
    appended bytes since the previous call are read. Otherwise a fresh
    state is constructed and a full parse runs.
    """
    log_dir = Path(log_dir)
    output_path = Path(output_path) if output_path else log_dir / "batch_metrics.png"

    if state is None:
        state = LogState(log_dir=log_dir)
    state.refresh()
    if not state.has_data():
        return False

    pf_agg = aggregate_prefill(state.prefill_files.values())
    dc_agg = aggregate_decode(state.decode_files.values())
    render_plot(pf_agg, dc_agg, output_path, title_prefix=get_run_title(log_dir), downsample=downsample)
    return True
