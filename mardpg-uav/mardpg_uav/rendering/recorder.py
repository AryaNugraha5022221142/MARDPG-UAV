"""Video + frame recording.

Task 3 ("Restore MP4 recording") root cause: the old pipeline relied on
matplotlib.animation.FuncAnimation(...).save(writer='ffmpeg'), wrapped the
whole thing in a bare `except Exception: pass`, and rebuilt the figure every
frame. If ffmpeg wasn't wired into matplotlib the save raised, the exception
was swallowed, and *no file and no error* ever appeared -- the exact
"videos are never written" symptom.

This module writes frames explicitly through OpenCV's VideoWriter (which ships
its own ffmpeg), verifies the writer opened, verifies the file is non-empty on
close, and raises/logs loudly on failure instead of swallowing it. An imageio
path is used as a fallback when OpenCV can't open a codec.

The recorder consumes RGB frames (as produced by SceneFrameRenderer) and is
codec/fps/resolution/filename-configurable per the RenderConfig.
"""

from __future__ import annotations

import os
import logging

import numpy as np

log = logging.getLogger("mardpg.render")


class VideoRecorder:
    """Write a sequence of RGB frames to an MP4 (or fall back to GIF)."""

    def __init__(self, out_path: str, fps: int = 20, codec: str = "mp4v",
                 frame_size=None):
        self.out_path = out_path
        self.fps = int(fps)
        self.codec = codec
        self.frame_size = frame_size  # (w, h); inferred from first frame if None
        self._writer = None
        self._backend = None
        self._imageio_frames = None
        self.n_written = 0
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # -- lifecycle ---------------------------------------------------------
    def _open(self, frame_size):
        self.frame_size = frame_size
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, frame_size)
            if writer.isOpened():
                self._writer = writer
                self._backend = "opencv"
                log.info("VideoRecorder: opencv writer opened -> %s (%dx%d @ %dfps, %s)",
                         self.out_path, frame_size[0], frame_size[1], self.fps, self.codec)
                return
            writer.release()
            log.warning("opencv could not open codec %r; trying imageio fallback.", self.codec)
        except Exception as e:  # pragma: no cover - environment specific
            log.warning("opencv unavailable (%s); trying imageio fallback.", e)

        # Fallback: buffer frames and hand them to imageio/ffmpeg on close.
        self._backend = "imageio"
        self._imageio_frames = []
        log.info("VideoRecorder: using imageio backend -> %s", self.out_path)

    def add_frame(self, rgb: np.ndarray):
        """Append one RGB (H, W, 3) uint8 frame."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        if self._backend is None:
            self._open((w, h))
        if (w, h) != self.frame_size:
            # Guard against a size drift that would corrupt the stream.
            import numpy as _np
            raise ValueError(
                f"frame size {(w, h)} != recorder size {self.frame_size}; "
                "all frames of one video must share a size.")
        if self._backend == "opencv":
            import cv2
            self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            self._imageio_frames.append(rgb)
        self.n_written += 1

    def close(self) -> str:
        """Finalise the file and return the actual output path written.

        Raises RuntimeError if nothing usable was produced -- so a failed
        recording is loud, never silent.
        """
        final_path = self.out_path
        if self._backend == "opencv" and self._writer is not None:
            self._writer.release()
            self._writer = None
        elif self._backend == "imageio":
            final_path = self._flush_imageio()

        if self.n_written == 0:
            raise RuntimeError(f"VideoRecorder wrote 0 frames to {self.out_path}")
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise RuntimeError(
                f"VideoRecorder produced no/empty file at {final_path} "
                f"(backend={self._backend}, frames={self.n_written})")
        log.info("VideoRecorder: wrote %d frames -> %s (%d bytes)",
                 self.n_written, final_path, os.path.getsize(final_path))
        return final_path

    def _flush_imageio(self) -> str:
        try:
            import imageio.v2 as imageio
        except ImportError:
            log.error("Cannot save video because imageio is not installed.")
            return ""

        try:
            imageio.mimsave(self.out_path, self._imageio_frames, fps=self.fps)
            return self.out_path
        except Exception as e:
            gif_path = os.path.splitext(self.out_path)[0] + ".gif"
            log.warning("imageio mp4 failed (%s); writing GIF %s instead.", e, gif_path)
            try:
                imageio.mimsave(gif_path, self._imageio_frames, fps=self.fps)
                return gif_path
            except Exception as e2:
                log.error("imageio gif fallback also failed: %s", e2)
                return ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # On error we still release the writer, but let the exception propagate.
        if self._backend == "opencv" and self._writer is not None:
            self._writer.release()
            self._writer = None


def save_frames_as_png(frames, out_dir: str, prefix: str = "frame", stride: int = 1):
    """Dump selected RGB frames as PNGs (Task 4: save_frames / every-N-frames)."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for idx, rgb in enumerate(frames):
        if idx % max(1, stride) != 0:
            continue
        p = os.path.join(out_dir, f"{prefix}_{idx:05d}.png")
        cv2.imwrite(p, cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2BGR))
        written.append(p)
    log.info("save_frames_as_png: wrote %d PNGs to %s (stride=%d)", len(written), out_dir, stride)
    return written
