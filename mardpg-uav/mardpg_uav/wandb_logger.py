import os
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class WandbLogger:
    def __init__(self, use_wandb: bool, project: str, config: Dict[str, Any], name: Optional[str] = None):
        self.use_wandb = bool(use_wandb)
        self._wandb = None
                
        if self.use_wandb:
            try:
                import wandb
                self._wandb = wandb
                if wandb.run is None:
                    has_creds = bool(
                        os.environ.get("WANDB_API_KEY")
                        or os.environ.get("WANDB_MODE")
                        or getattr(wandb.api, "api_key", None)
                    )
                    if not has_creds:
                        os.environ.setdefault("WANDB_MODE", "offline")
                    wandb.init(project=project, config=config, name=name, settings=wandb.Settings(init_timeout=30))
            except Exception as e:
                log.warning(f"Wandb init failed ({e}). Disabling wandb logging.")
                self.use_wandb = False
                self._wandb = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.use_wandb and self._wandb and getattr(self._wandb, "run", None):
            if step is not None:
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics)

    def finish(self):
        if self.use_wandb and self._wandb and getattr(self._wandb, "run", None):
            self._wandb.finish()

    def log_video(self, key: str, path: str, step: Optional[int] = None, fps: int = 4):
        if self.use_wandb and self._wandb and getattr(self._wandb, "run", None) and os.path.exists(path):
            self.log({key: self._wandb.Video(path, fps=fps, format="mp4")}, step=step)

    def log_image(self, key: str, path: str, step: Optional[int] = None):
        if self.use_wandb and self._wandb and getattr(self._wandb, "run", None) and os.path.exists(path):
            self.log({key: self._wandb.Image(path)}, step=step)

    def log_media(self, media_dict: Dict[str, Any], prefix: str = "", step: Optional[int] = None, fps: int = 4):
        if not (self.use_wandb and self._wandb and getattr(self._wandb, "run", None)):
            return
        to_log = {}
        for k, v in media_dict.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, str) and os.path.exists(v):
                if v.endswith(".mp4"):
                    to_log[key] = self._wandb.Video(v, fps=fps, format="mp4")
                elif v.endswith((".png", ".jpg", ".jpeg")):
                    to_log[key] = self._wandb.Image(v)
        if to_log:
            self.log(to_log, step=step)

    def log_table(self, key: str, dataframe: Any, step: Optional[int] = None):
        if self.use_wandb and self._wandb and getattr(self._wandb, "run", None):
            self.log({key: self._wandb.Table(dataframe=dataframe)}, step=step)

    def log_artifact(self, name: str, paths: list, type: str = "dataset", description: str = ""):
        if self.use_wandb and self._wandb and getattr(self._wandb, "run", None):
            art = self._wandb.Artifact(name, type=type, description=description)
            for p in paths:
                if os.path.exists(p):
                    if os.path.isdir(p):
                        art.add_dir(p)
                    else:
                        art.add_file(p)
            self._wandb.run.log_artifact(art)
