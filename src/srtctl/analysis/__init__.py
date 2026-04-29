# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analysis utilities reusable from the orchestrator and standalone CLIs.

This subpackage contains:

- ``batch_metrics``: parse SGLang prefill/decode worker logs and render
  per-cluster aggregated time-series PNGs. Used both as a post-mortem
  CLI (``plot_batch_metrics.py`` at the repo root) and as the core of
  the in-flight live snapshotter (``live_metrics``).
- ``live_metrics``: a background snapshotter that periodically refreshes
  ``batch_metrics.png`` while a benchmark is running.
"""
