"""Offline unit tests for Ogun's prompt line editing (ogun_core/lineedit.py).

Covers: history path resolution ($OGUN_HISTORY_FILE override, ~/.ogun_history
default), setup against a fake readline module (history length, loading an
existing history file, atexit registration, idempotence), graceful behaviour
when readline is missing or the history file is unreadable, and saving
history (parent dirs created, owner-only permissions).

No terminal is involved: the readline module is faked at the module boundary.
Standalone runnable: python tests/test_lineedit.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

OGUN_ROOT = Path(__file__).resolve().parents[1]
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

from ogun_core import lineedit


class FakeReadline:
    """Records the calls lineedit makes and can be told to fail on read."""

    def __init__(self, fail_read: bool = False):
        self.calls = []
        self.fail_read = fail_read

    def set_history_length(self, length):
        self.calls.append(("set_history_length", length))

    def read_history_file(self, path):
        self.calls.append(("read_history_file", path))
        if self.fail_read:
            raise OSError("corrupt history")

    def write_history_file(self, path):
        self.calls.append(("write_history_file", path))
        Path(path).write_text("fake history\n", encoding="utf-8")


def _reset(fake):
    """Point lineedit at a fake readline and clear its one-time setup flag."""
    lineedit._setup_done = False
    return patch.object(lineedit, "readline", fake)


def test_history_path_default_and_override():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(lineedit.HISTORY_ENV_VAR, None)
        assert lineedit.history_path() == Path.home() / ".ogun_history"
    with patch.dict(os.environ, {lineedit.HISTORY_ENV_VAR: "/tmp/elsewhere/hist"}):
        assert lineedit.history_path() == Path("/tmp/elsewhere/hist")
    with patch.dict(os.environ, {lineedit.HISTORY_ENV_VAR: "  "}):
        # A blank override is ignored, not treated as a relative path.
        assert lineedit.history_path() == Path.home() / ".ogun_history"


def test_setup_loads_history_and_registers_save():
    fake = FakeReadline()
    with tempfile.TemporaryDirectory() as td:
        hist = Path(td) / "hist"
        hist.write_text("earlier entry\n", encoding="utf-8")
        registered = []
        with _reset(fake), \
             patch.dict(os.environ, {lineedit.HISTORY_ENV_VAR: str(hist)}), \
             patch.object(lineedit.atexit, "register", registered.append):
            assert lineedit.setup_history() is True
            assert lineedit.readline_available() is True
            assert ("set_history_length", lineedit.HISTORY_LENGTH) in fake.calls
            assert ("read_history_file", str(hist)) in fake.calls
            assert registered == [lineedit.save_history]
            # Idempotent: a second call must not re-read or re-register.
            fake.calls.clear()
            assert lineedit.setup_history() is True
            assert fake.calls == []


def test_setup_without_existing_history_file():
    fake = FakeReadline()
    with tempfile.TemporaryDirectory() as td:
        hist = Path(td) / "not-there-yet"
        with _reset(fake), \
             patch.dict(os.environ, {lineedit.HISTORY_ENV_VAR: str(hist)}), \
             patch.object(lineedit.atexit, "register", lambda fn: None):
            assert lineedit.setup_history() is True
            assert not any(name == "read_history_file" for name, _ in fake.calls)


def test_setup_tolerates_unreadable_history():
    fake = FakeReadline(fail_read=True)
    with tempfile.TemporaryDirectory() as td:
        hist = Path(td) / "hist"
        hist.write_text("x\n", encoding="utf-8")
        with _reset(fake), \
             patch.dict(os.environ, {lineedit.HISTORY_ENV_VAR: str(hist)}), \
             patch.object(lineedit.atexit, "register", lambda fn: None):
            # Editing must still come up even when the history file is bad.
            assert lineedit.setup_history() is True


def test_setup_without_readline_module():
    lineedit._setup_done = False
    with patch.object(lineedit, "readline", None):
        assert lineedit.readline_available() is False
        assert lineedit.setup_history() is False
        lineedit.save_history()  # must be a silent no-op, not a crash


def test_save_history_writes_owner_only_file():
    fake = FakeReadline()
    with tempfile.TemporaryDirectory() as td:
        hist = Path(td) / "nested" / "hist"
        with _reset(fake), patch.dict(os.environ, {lineedit.HISTORY_ENV_VAR: str(hist)}):
            lineedit.save_history()
            assert ("write_history_file", str(hist)) in fake.calls
            assert hist.exists()
            if os.name == "posix":
                assert (hist.stat().st_mode & 0o777) == 0o600


def _scripted(lines):
    """An input_fn that replays `lines`, then raises EOFError like Ctrl+D."""
    it = iter(lines)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _input


def test_read_message_single_line_passthrough():
    assert lineedit.read_message("p", input_fn=_scripted(["hello"])) == "hello"
    # Not stripped: the caller owns that decision, as with plain input().
    assert lineedit.read_message("p", input_fn=_scripted(["  spaced  "])) == "  spaced  "
    # A line merely CONTAINING the fence later on is not a fence opener.
    assert lineedit.read_message("p", input_fn=_scripted(['say """ hi'])) == 'say """ hi'


def test_read_message_paste_collects_until_lone_dot():
    msg = lineedit.read_message("p", input_fn=_scripted(
        ["/paste", "#!/bin/sh", "  indented", "", "echo .", ".", "never read"]))
    # Markers excluded, content verbatim (indent + blank line kept), and a
    # line like `echo .` must NOT terminate the collection early.
    assert msg == "#!/bin/sh\n  indented\n\necho ."


def test_read_message_paste_eof_ends_collection():
    msg = lineedit.read_message("p", input_fn=_scripted(["/paste", "a", "b"]))
    assert msg == "a\nb"


def test_read_message_fence_same_line():
    assert lineedit.read_message("p", input_fn=_scripted(['"""echo hi"""'])) == "echo hi"


def test_read_message_fence_multiline():
    msg = lineedit.read_message("p", input_fn=_scripted(
        ['"""', "#!/bin/sh", 'echo "$1"', '"""', "never read"]))
    assert msg == '#!/bin/sh\necho "$1"'
    # Content may share the fence lines themselves.
    msg = lineedit.read_message("p", input_fn=_scripted(['"""first', "mid", 'last"""']))
    assert msg == "first\nmid\nlast"


def test_read_message_fence_unterminated_eof():
    msg = lineedit.read_message("p", input_fn=_scripted(['"""', "x"]))
    assert msg == "x"


def test_read_message_first_line_eof_raises():
    try:
        lineedit.read_message("p", input_fn=_scripted([]))
    except EOFError:
        pass
    else:
        raise AssertionError("EOFError on the first line must propagate")


def test_real_readline_is_present_on_this_platform():
    # The whole feature rests on the stdlib module existing; on platforms
    # where it does not, setup_history() must already have returned False
    # above. Here we only assert the module-level contract.
    assert lineedit.readline_available() == (lineedit.readline is not None)


def main() -> int:
    tests = [
        test_history_path_default_and_override,
        test_setup_loads_history_and_registers_save,
        test_setup_without_existing_history_file,
        test_setup_tolerates_unreadable_history,
        test_setup_without_readline_module,
        test_save_history_writes_owner_only_file,
        test_read_message_single_line_passthrough,
        test_read_message_paste_collects_until_lone_dot,
        test_read_message_paste_eof_ends_collection,
        test_read_message_fence_same_line,
        test_read_message_fence_multiline,
        test_read_message_fence_unterminated_eof,
        test_read_message_first_line_eof_raises,
        test_real_readline_is_present_on_this_platform,
    ]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as ae:
        print(f"TEST ASSERTION FAILED: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"TEST RUNTIME ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
