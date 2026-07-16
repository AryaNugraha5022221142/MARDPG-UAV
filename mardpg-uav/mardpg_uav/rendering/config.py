"""Typed view over the `render:` block of the YAML config.

Keeping this in one dataclass means the rest of the codebase never reaches
into raw dicts with `.get(...)` scattered everywhere, and every default lives
in exactly one place. `RenderConfig.from_config(cfg)` tolerates a config that
predates the render block (returns all-defaults) so old checkpoints/configs
keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class RenderConfig:
    enable_render: bool = False
    realtime_render: bool = False
    record_video: bool = True
    save_png: bool = True
    save_frames: bool = False
    frame_stride: int = 1
    video_fps: int = 20
    video_codec: str = "mp4v"
    render_backend: str = "auto"
    render_resolution: List[int] = field(default_factory=lambda: [1280, 960])
    image_dpi: int = 200
    wandb_log_media: bool = True
    output_directory: str = "eval_results"

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "RenderConfig":
        """Build from a full config dict (reads the `render` sub-dict)."""
        block = {}
        if isinstance(cfg, dict):
            block = cfg.get("render", {}) or {}
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in block.items() if k in known}
        return cls(**clean)

    def merged_with_cli(self, **overrides) -> "RenderConfig":
        """Return a copy with any non-None keyword overrides applied.

        CLI flags pass their value here; a value of None means "not set on the
        command line, keep the config value". This is what lets a single config
        drive both cloud and local runs without code changes.
        """
        data = asdict(self)
        for k, v in overrides.items():
            if v is not None and k in data:
                data[k] = v
        return RenderConfig(**data)

    @property
    def width(self) -> int:
        return int(self.render_resolution[0])

    @property
    def height(self) -> int:
        return int(self.render_resolution[1])

    def any_media_requested(self) -> bool:
        """True if the run should produce *any* file/media output."""
        return self.enable_render and (
            self.record_video or self.save_png or self.save_frames
        )
