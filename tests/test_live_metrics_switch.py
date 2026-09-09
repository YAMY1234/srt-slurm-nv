# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-level on/off switch for the live batch-metrics snapshotter."""

import threading

from srtctl.analysis import live_metrics
from srtctl.core.schema import LiveMetricsConfig, TelemetryConfig


def test_recipe_block_off_never_consults_cluster_config(monkeypatch, tmp_path):
    def boom():  # pragma: no cover - must not be called
        raise AssertionError("cluster config consulted although recipe block is set")

    monkeypatch.setattr("srtctl.core.config.load_cluster_config", boom)
    cfg = LiveMetricsConfig(enabled=False)
    assert live_metrics.try_start_snapshotter(tmp_path, threading.Event(), recipe_config=cfg) is None


def test_recipe_block_on_starts_snapshotter_with_recipe_knobs(tmp_path):
    cfg = LiveMetricsConfig(enabled=True, interval_seconds=7, downsample=3)
    stop = threading.Event()
    snap = live_metrics.try_start_snapshotter(tmp_path, stop, recipe_config=cfg)
    try:
        assert snap is not None
        assert snap._params.interval_seconds == 7
        assert snap._params.downsample == 3
    finally:
        if snap is not None:
            snap.stop()


def test_unset_recipe_block_falls_back_to_cluster_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "srtctl.core.config.load_cluster_config",
        lambda: {"telemetry": {"live_metrics": {"enabled": True, "interval_seconds": 9}}},
    )
    stop = threading.Event()
    snap = live_metrics.try_start_snapshotter(tmp_path, stop, recipe_config=None)
    try:
        assert snap is not None
        assert snap._params.interval_seconds == 9
    finally:
        if snap is not None:
            snap.stop()


def test_telemetry_schema_accepts_live_metrics_without_enabling_power():
    loaded = TelemetryConfig.Schema().load({"live_metrics": {"enabled": True, "interval_seconds": 60}})
    assert loaded.enabled is False
    assert loaded.live_metrics == LiveMetricsConfig(enabled=True, interval_seconds=60, downsample=1)
