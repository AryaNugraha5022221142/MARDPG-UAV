import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force headless so it mirrors a GCP VM run.
from mardpg_uav.rendering.backend import select_backend
print("backend:", select_backend("Agg", want_interactive=False))

from dataclasses import dataclass
from mardpg_uav.rendering.frames import SceneFrameRenderer
from mardpg_uav.rendering.recorder import VideoRecorder, save_frames_as_png
from mardpg_uav.rendering.static_plots import plot_trajectory_3d, plot_trajectory_top_down, plot_summary
from mardpg_uav.rendering.config import RenderConfig
from mardpg_uav.rendering.media import generate_episode_media


@dataclass
class Ob:
    type: str
    position: np.ndarray
    size: np.ndarray
    velocity: np.ndarray = None


class FakeEnv:
    """Just enough surface for scene drawing: .obstacles (last 6 = walls) and .goals."""
    def __init__(self, n_agents, env_size):
        self.n_agents = n_agents
        ex, ey, ez = env_size
        cyl = [Ob("cylinder", np.array([30., 40., 10.]), np.array([4.0, 20.0])),
               Ob("box", np.array([60., 55., 8.]), np.array([5.0, 5.0, 8.0]))]
        walls = [Ob("box", np.array([ex/2, ey/2, ez/2]), np.array([1., 1., 1.]))] * 6
        self.obstacles = cyl + walls
        self.goals = np.array([[ex*0.8, ey*0.8, ez*0.5],
                               [ex*0.2, ey*0.7, ez*0.4],
                               [ex*0.6, ey*0.2, ez*0.6]])[:n_agents]


def make_bundle(n_agents, env_size, T=60, with_dyn=True):
    ex, ey, ez = env_size
    starts = np.array([[ex*0.1, ey*0.1, ez*0.5],
                       [ex*0.9, ey*0.2, ez*0.4],
                       [ex*0.3, ey*0.9, ez*0.6]])[:n_agents]
    goals = np.array([[ex*0.8, ey*0.8, ez*0.5],
                      [ex*0.2, ey*0.7, ez*0.4],
                      [ex*0.6, ey*0.2, ez*0.6]])[:n_agents]
    ts = np.linspace(0, 1, T)[:, None, None]
    path = starts[None] + ts * (goals - starts)[None]  # [T, N, 3] straight lines
    path += 2.0 * np.sin(ts * 6 * np.pi) * np.array([[0, 0, 1.0]])
    dyn_path = None
    dyn_r = []
    if with_dyn:
        d0 = np.array([[ex*0.5, ey*0.5, ez*0.5]])
        dyn_path = d0[None] + (np.sin(ts * 4 * np.pi) * 10) * np.array([[1.0, 0, 0]])
        dyn_path = dyn_path.reshape(T, 1, 3)
        dyn_r = [2.5]
    return dict(path=path, goals=goals,
                reached=np.array([True, False, True])[:n_agents],
                collided=np.array([False, True, False])[:n_agents],
                dyn_path=dyn_path, dyn_r=dyn_r)


def main():
    outdir = "/tmp/render_test_out"
    os.makedirs(outdir, exist_ok=True)
    n_agents, env_size = 3, [100.0, 100.0, 60.0]
    env_cfg = {"n_agents": n_agents, "env_size": env_size,
               "inter_uav_min_dist": 1.0, "goal_threshold": 1.0}
    env = FakeEnv(n_agents, env_size)
    rnd = make_bundle(n_agents, env_size, T=48, with_dyn=True)

    # 1) SceneFrameRenderer: reuse artists, produce RGB frames + timing.
    import time
    fr = SceneFrameRenderer(env, env_cfg, rnd, width=640, height=480, dpi=100, tail=40)
    t0 = time.time()
    frames = [fr.render_frame(t) for t in range(fr.T)]
    dt = time.time() - t0
    fr.close()
    assert frames[0].shape[2] == 3 and frames[0].dtype == np.uint8
    fps = fr.T / dt
    print(f"[frames] {fr.T} frames {frames[0].shape} in {dt:.2f}s -> {fps:.1f} render-fps")

    # 2) VideoRecorder MP4
    vpath = os.path.join(outdir, "test_best.mp4")
    rec = VideoRecorder(vpath, fps=20, codec="mp4v")
    for f in frames:
        rec.add_frame(f)
    final = rec.close()
    print(f"[video] wrote {final} ({os.path.getsize(final)} bytes, {rec.n_written} frames)")

    # 3) per-frame PNG dump (every 10th)
    png_dir = os.path.join(outdir, "test_frames")
    pngs = save_frames_as_png(frames, png_dir, prefix="f", stride=10)
    print(f"[frames-png] {len(pngs)} PNGs")

    # 4) static publication plots
    p3d = plot_trajectory_3d(env, env_cfg, rnd, "unit-test 3d", os.path.join(outdir, "static_3d.png"), dpi=120)
    p2d = plot_trajectory_top_down(env, env_cfg, rnd, "unit-test topdown", os.path.join(outdir, "static_topdown.png"), dpi=120)
    psum = plot_summary(env, env_cfg, rnd, "unit-test", os.path.join(outdir, "static_summary.png"), dpi=110)
    for p in (p3d, p2d, psum):
        assert os.path.getsize(p) > 0
    print(f"[static] wrote 3 PNGs, sizes: "
          f"{os.path.getsize(p3d)}, {os.path.getsize(p2d)}, {os.path.getsize(psum)}")

    # 5) orchestrator, exactly as the eval script calls it
    rcfg = RenderConfig(enable_render=True, record_video=True, save_png=True,
                        save_frames=True, frame_stride=8, video_fps=20,
                        render_resolution=[640, 480], image_dpi=120,
                        output_directory=outdir)
    produced = generate_episode_media(env, env_cfg, rnd, rcfg,
                                      tag="orch_best", title="orchestrator test", outdir=outdir)
    print(f"[orchestrator] produced keys: {sorted(produced)}")
    for k, v in produced.items():
        if k == "frames_dir":
            assert os.path.isdir(v) and len(os.listdir(v)) > 0
        else:
            assert os.path.exists(v) and os.path.getsize(v) > 0
    print("[orchestrator] all outputs verified non-empty")

    # 6) headless LiveRenderer must be a safe no-op
    from mardpg_uav.rendering.live import LiveRenderer
    lr = LiveRenderer(env, env_cfg)
    assert lr.enabled is False
    lr.reset(env)
    lr.close()
    print("[live] headless no-op OK")

    print("\nALL RENDER TESTS PASSED")


if __name__ == "__main__":
    main()
