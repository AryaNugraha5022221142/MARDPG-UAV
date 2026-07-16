"""Rendering subsystem for MARDPG-UAV.

This package is the single home for everything that turns a rollout into
pixels: backend selection, the real-time viewer, the frame renderer that
feeds the video recorder, and the publication-quality static plots.

Design goals (see docs/RENDERING.md):
  * One backend decision, made once, honouring headless/cloud environments.
  * Reuse matplotlib figures and artists across scenes and frames instead of
    rebuilding them every timestep (the root cause of the old slowness).
  * Correctness is independent of torch/checkpoints -- everything operates on
    plain numpy trajectory arrays, so it can be tested in isolation.
"""

from .config import RenderConfig
from .backend import select_backend
from .live import LiveRenderer
from .frames import SceneFrameRenderer
from .static_plots import (
    plot_trajectory_3d,
    plot_trajectory_top_down,
    plot_summary,
)

__all__ = [
    "RenderConfig",
    "select_backend",
    "LiveRenderer",
    "SceneFrameRenderer",
    "plot_trajectory_3d",
    "plot_trajectory_top_down",
    "plot_summary",
]
