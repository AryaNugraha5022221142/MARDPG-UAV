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
                    wandb.init(project=project, config=config, name=name, settings=wandb.Settings(init_timeout=15))
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
