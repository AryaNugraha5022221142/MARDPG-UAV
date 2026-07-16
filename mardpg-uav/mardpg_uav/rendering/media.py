"""High-level media generation for one evaluation episode.

This is the single entry point the eval script calls after it has a render
bundle (`rnd`). It honours a RenderConfig and produces, as requested:
  * an MP4 video (via SceneFrameRenderer -> VideoRecorder),
  * optional per-frame PNGs,
  * publication PNGs (3D + top-down),
and returns a dict of {label: path} for everything written, which the W&B
logger can then upload. Failures are logged loudly and surfaced, never
swallowed -- the caller decides whether to continue.
"""

from __future__ import annotations

import os
import logging

from .config import RenderConfig
from .frames import SceneFrameRenderer
from .recorder import VideoRecorder, save_frames_as_png
from .static_plots import plot_trajectory_3d, plot_trajectory_top_down

log = logging.getLogger("mardpg.render")


def generate_episode_media(env, env_cfg, rnd, rcfg: RenderConfig,
                           tag: str, title: str, outdir: str) -> dict:
    """Produce all requested media for one episode.

    Args:
        env, env_cfg: environment + its config (for geometry).
        rnd: render bundle (path/goals/reached/collided/dyn_path/dyn_r).
        rcfg: RenderConfig controlling what to produce.
        tag: filename-safe identifier, e.g. 'best_MARDPG_s2_ood_dense_25'.
        title: human-readable figure title.
        outdir: directory to write into (created if missing).

    Returns:
        dict mapping a media label to its file path, e.g.
        {'video': '.../best_...mp4', 'png_3d': '...', 'png_topdown': '...'}.
    """
    os.makedirs(outdir, exist_ok=True)
    produced = {}

    if rcfg.save_png:
        p3d = os.path.join(outdir, f"{tag}_3d.png")
        p2d = os.path.join(outdir, f"{tag}_topdown.png")
        plot_trajectory_3d(env, env_cfg, rnd, title, p3d, dpi=rcfg.image_dpi)
        plot_trajectory_top_down(env, env_cfg, rnd, title, p2d, dpi=rcfg.image_dpi)
        produced["png_3d"] = p3d
        produced["png_topdown"] = p2d

    if rcfg.record_video or rcfg.save_frames:
        renderer = SceneFrameRenderer(
            env, env_cfg, rnd,
            width=rcfg.width, height=rcfg.height, dpi=100)
        try:
            frames = [renderer.render_frame(t) for t in range(renderer.T)]
        finally:
            renderer.close()

        if rcfg.record_video:
            vid_path = os.path.join(outdir, f"{tag}.mp4")
            rec = VideoRecorder(vid_path, fps=rcfg.video_fps, codec=rcfg.video_codec)
            for f in frames:
                rec.add_frame(f)
            produced["video"] = rec.close()  # actual path (mp4 or gif fallback)

        if rcfg.save_frames:
            frame_dir = os.path.join(outdir, f"{tag}_frames")
            save_frames_as_png(frames, frame_dir, prefix=tag, stride=rcfg.frame_stride)
            produced["frames_dir"] = frame_dir

    log.info("generate_episode_media[%s]: produced %s", tag, list(produced.keys()))
    return produced
