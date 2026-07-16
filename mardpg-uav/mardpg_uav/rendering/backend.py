"""Matplotlib backend selection, made once and made explicit.

The old code toggled `matplotlib.use('Agg')` by editing a commented line by
hand -- which is exactly why rendering "silently failed" on the VM (an
interactive backend with no display raises or hangs) and why local realtime
froze (Agg can't show a window). This module makes the decision deterministic
and logged.

Rules:
  * backend == 'Agg'            -> force headless (file output only).
  * backend interactive name    -> use it verbatim (e.g. 'TkAgg', 'QtAgg').
  * backend == 'auto' (default) -> interactive iff a display is available AND
    realtime rendering was requested; otherwise Agg.

A "display is available" means $DISPLAY on Linux, or being on Windows/macOS
where a GUI backend normally works. On a headless GCP VM there is no $DISPLAY,
so we fall back to Agg and never try to open a window.
"""

from __future__ import annotations

import os
import sys
import logging

log = logging.getLogger("mardpg.render")

_INTERACTIVE_CANDIDATES = ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX")


def _display_available() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    # Linux / other unix: need an X or Wayland display.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _try_interactive() -> str:
    import matplotlib
    for name in _INTERACTIVE_CANDIDATES:
        try:
            matplotlib.use(name, force=True)
            return name
        except Exception:
            continue
    matplotlib.use("Agg", force=True)
    return "Agg"


def select_backend(requested: str = "auto", want_interactive: bool = False) -> str:
    """Set and return the matplotlib backend actually in effect.

    Args:
        requested: value of render_backend from config ('auto'/'Agg'/'TkAgg'...).
        want_interactive: True if the caller wants a live on-screen window.

    Returns:
        The backend name now active. Never raises: on any failure it falls back
        to 'Agg' so file-based rendering always works.
    """
    import matplotlib

    req = (requested or "auto").strip()

    if req.lower() == "agg":
        matplotlib.use("Agg", force=True)
        chosen = "Agg"
    elif req.lower() != "auto":
        # Explicit interactive backend requested.
        try:
            matplotlib.use(req, force=True)
            chosen = req
        except Exception as e:  # pragma: no cover - environment specific
            log.warning("Requested backend %r unavailable (%s); using Agg.", req, e)
            matplotlib.use("Agg", force=True)
            chosen = "Agg"
    else:  # auto
        if want_interactive and _display_available():
            chosen = _try_interactive()
            if chosen == "Agg":
                log.warning(
                    "Realtime render requested but no interactive backend could "
                    "be initialised; running headless (Agg)."
                )
        else:
            matplotlib.use("Agg", force=True)
            chosen = "Agg"
            if want_interactive and not _display_available():
                log.warning(
                    "Realtime render requested but no display detected "
                    "(headless/cloud). Falling back to Agg -- video/PNG still work."
                )

    log.info("matplotlib backend: %s (requested=%s, interactive=%s)",
             chosen, requested, want_interactive)
    return chosen


def is_interactive_backend() -> bool:
    import matplotlib
    return matplotlib.get_backend().lower() != "agg"
