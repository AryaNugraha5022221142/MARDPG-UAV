from __future__ import annotations

import os
import logging

log = logging.getLogger("mardpg.wandb")


class WandbLogger:
    def __init__(self, use_wandb, project, config, name):
        self.use_wandb = bool(use_wandb)
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb
                self._wandb = wandb
                if wandb.run is None:
                    wandb.init(project=project, config=config, name=name)
            except Exception as e:
                log.warning("Wandb init failed (%s). Disabling wandb logging.", e)
                self.use_wandb = False
                self._wandb = None

    # -- scalars -----------------------------------------------------------
    def log(self, data, step=None):
        if self.use_wandb:
            self._wandb.log(data, step=step)

    # -- media -------------------------------------------------------------
    def log_video(self, key, path, fps=20, step=None):
        """Upload a video. Handles the mp4->gif fallback path transparently."""
        if not self.use_wandb:
            return False
        if not path or not os.path.exists(path):
            log.warning("wandb.log_video: file missing, skipping: %s", path)
            return False
        fmt = "gif" if path.lower().endswith(".gif") else "mp4"
        self._wandb.log({key: self._wandb.Video(path, fps=fps, format=fmt)}, step=step)
        return True

    def log_image(self, key, path, step=None):
        if not self.use_wandb:
            return False
        if not path or not os.path.exists(path):
            log.warning("wandb.log_image: file missing, skipping: %s", path)
            return False
        self._wandb.log({key: self._wandb.Image(path)}, step=step)
        return True

    def log_media(self, produced: dict, prefix: str, step=None, fps=20):
        """Upload a dict of media produced by rendering.generate_episode_media.

        Keys understood: 'video', 'png_3d', 'png_topdown'. Unknown keys (e.g.
        'frames_dir') are ignored for logging but not an error.
        """
        if not self.use_wandb:
            return
        if "video" in produced:
            self.log_video(f"{prefix}/video", produced["video"], fps=fps, step=step)
        if "png_3d" in produced:
            self.log_image(f"{prefix}/traj_3d", produced["png_3d"], step=step)
        if "png_topdown" in produced:
            self.log_image(f"{prefix}/traj_topdown", produced["png_topdown"], step=step)

    def log_table(self, key, dataframe, step=None):
        if self.use_wandb:
            self._wandb.log({key: self._wandb.Table(dataframe=dataframe)}, step=step)

    # -- artifacts ---------------------------------------------------------
    def log_artifact(self, name, artifact_type, paths, metadata=None):
        """Bundle files/dirs into a versioned W&B Artifact and upload it."""
        if not self.use_wandb:
            return False
        art = self._wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
        added = 0
        for p in paths:
            if not p or not os.path.exists(p):
                log.warning("wandb.log_artifact: path missing, skipping: %s", p)
                continue
            if os.path.isdir(p):
                art.add_dir(p)
            else:
                art.add_file(p)
            added += 1
        if added == 0:
            log.warning("wandb.log_artifact(%s): no valid files, not logging.", name)
            return False
        self._wandb.log_artifact(art)
        return True

    def finish(self):
        if self.use_wandb and self._wandb.run is not None:
            self._wandb.finish()
