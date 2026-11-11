"""Bash-style line editing and history for Ogun's interactive prompt.

Wires Python's stdlib ``readline`` (the library bash itself is built on) into
the chat loop's ``input()`` calls. Once set up, the prompt supports:

- Left/Right arrows (and Home/End, Ctrl+A / Ctrl+E) to move within the line
  being typed, so a message can be fixed before it is sent
- Alt+B / Alt+F to move by word, Ctrl+W / Ctrl+U / Ctrl+K to delete by
  word / line-start / line-end
- Up/Down arrows to walk back through earlier entries, exactly like bash
- Ctrl+R reverse search through those entries

Entries persist across sessions in ``~/.ogun_history`` (override the location
with ``$OGUN_HISTORY_FILE``), capped at HISTORY_LENGTH entries and written
with owner-only permissions, since prompts can contain sensitive text.

Everything degrades silently: a Python build without the readline module, or
any history-file problem, simply keeps the old plain ``input()`` behaviour.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path

try:
    import readline
except ImportError:  # minimal or embedded Python builds may lack the module
    readline = None

HISTORY_ENV_VAR = "OGUN_HISTORY_FILE"
HISTORY_LENGTH = 1000

_setup_done = False


def history_path() -> Path:
    """Where prompt history lives: $OGUN_HISTORY_FILE, else ~/.ogun_history."""
    override = os.environ.get(HISTORY_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ogun_history"


def readline_available() -> bool:
    return readline is not None


def save_history() -> None:
    """Persist the in-memory history (registered with atexit); never raises."""
    if readline is None:
        return
    try:
        path = history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(path))
        os.chmod(path, 0o600)
    except Exception:
        pass


def setup_history() -> bool:
    """Load history and turn on readline editing for input().

    Returns True when line editing is active, False when the readline module
    is unavailable (the caller can then simply use plain input()). Idempotent:
    repeat calls change nothing and do not re-register the exit hook.
    """
    global _setup_done
    if readline is None:
        return False
    if _setup_done:
        return True
    _setup_done = True
    try:
        readline.set_history_length(HISTORY_LENGTH)
        path = history_path()
        if path.exists():
            readline.read_history_file(str(path))
    except Exception:
        # A corrupt or unreadable history file must not cost the user editing.
        pass
    atexit.register(save_history)
    return True
