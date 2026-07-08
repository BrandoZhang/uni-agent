"""Tests for per-run / per-session media workspace isolation.

Covers ``uni_agent.workspace.resolve_media_workspace`` — the pure helper
``UniAgentLoop`` uses to key each rollout's artifacts + cost ledger by a
``workspace_key`` (run_id for one-shot rollouts, a pinned session id for
resumable interactive tasks) so concurrent runs on a shared filesystem don't
collide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uni_agent.workspace import (
    compose_post_setup_cmd,
    resolve_media_workspace,
    should_gc_workspace,
    workspace_gc_command,
)


def test_no_base_is_a_noop():
    """Without MEDIA_WORKSPACE_BASE, isolation is off and env is untouched."""
    env = {"MEDIA_BACKEND": "mock", "MEDIA_WORKSPACE": "/pinned/ws"}
    out, ws = resolve_media_workspace(env, "run-123")
    assert ws is None
    assert out == env
    assert out is not env  # returns a copy, never mutates the caller's dict


def test_none_env_is_a_noop():
    out, ws = resolve_media_workspace(None, "run-123")
    assert ws is None
    assert out == {}


def test_base_keys_workspace_by_run_id():
    """One-shot rollout: workspace_key defaults to the per-run run_id."""
    out, ws = resolve_media_workspace({"MEDIA_WORKSPACE_BASE": "/tmp/base"}, "run-abc")
    assert ws == "/tmp/base/run-abc"
    assert out["MEDIA_WORKSPACE"] == "/tmp/base/run-abc"
    assert out["MEDIA_USAGE_LOG"] == "/tmp/base/run-abc/usage.jsonl"


def test_stable_key_overrides_run_id():
    """Resumable session: a pinned MEDIA_WORKSPACE_KEY wins over run_id, so the
    same task maps to the same dir across invocations (artifacts persist)."""
    env = {"MEDIA_WORKSPACE_BASE": "/tmp/base", "MEDIA_WORKSPACE_KEY": "session-7"}
    out_a, ws_a = resolve_media_workspace(env, "run-1")
    out_b, ws_b = resolve_media_workspace(env, "run-2")  # different run_id, same session
    assert ws_a == ws_b == "/tmp/base/session-7"
    assert out_a["MEDIA_USAGE_LOG"] == out_b["MEDIA_USAGE_LOG"]


def test_distinct_run_ids_get_distinct_workspaces():
    """Two concurrent one-shot rollouts never share a dir or ledger."""
    _, ws1 = resolve_media_workspace({"MEDIA_WORKSPACE_BASE": "/tmp/base"}, "run-1")
    _, ws2 = resolve_media_workspace({"MEDIA_WORKSPACE_BASE": "/tmp/base"}, "run-2")
    assert ws1 != ws2


def test_pinned_usage_log_is_respected():
    """An explicit MEDIA_USAGE_LOG is not overwritten (caller opted in)."""
    env = {"MEDIA_WORKSPACE_BASE": "/tmp/base", "MEDIA_USAGE_LOG": "/custom/ledger.jsonl"}
    out, ws = resolve_media_workspace(env, "run-x")
    assert out["MEDIA_USAGE_LOG"] == "/custom/ledger.jsonl"
    # ...but the workspace itself is still derived
    assert out["MEDIA_WORKSPACE"] == "/tmp/base/run-x"


def test_base_home_expansion():
    out, ws = resolve_media_workspace({"MEDIA_WORKSPACE_BASE": "~/media"}, "run-x")
    assert ws == str(Path("~/media").expanduser() / "run-x")
    assert "~" not in ws


def test_compose_post_setup_cmd_creates_and_cds():
    cmd = compose_post_setup_cmd(None, "/tmp/base/run-x")
    assert cmd == "mkdir -p /tmp/base/run-x && cd /tmp/base/run-x"


def test_compose_post_setup_cmd_preserves_existing():
    cmd = compose_post_setup_cmd("export FOO=1", "/tmp/base/run-x")
    assert cmd == "export FOO=1 && mkdir -p /tmp/base/run-x && cd /tmp/base/run-x"


def test_compose_post_setup_cmd_quotes_spaces():
    cmd = compose_post_setup_cmd(None, "/tmp/a b/run x")
    # shlex.quote wraps paths containing spaces so the shell sees one arg
    assert "'/tmp/a b/run x'" in cmd


# --------------------------------------------------------------------------
# run-end GC
# --------------------------------------------------------------------------


def test_gc_off_by_default():
    """No MEDIA_WORKSPACE_GC -> GC is off (artifacts preserved)."""
    assert should_gc_workspace({"MEDIA_WORKSPACE_BASE": "/b"}, "/b/run-x") is False


def test_gc_requires_a_derived_workspace():
    # even opted-in, nothing to reclaim if isolation wasn't active
    assert should_gc_workspace({"MEDIA_WORKSPACE_GC": "reclaim"}, None) is False


@pytest.mark.parametrize("value", ["reclaim", "1", "true", "TRUE", "yes", "on", " On "])
def test_gc_truthy_values_enable(value):
    assert should_gc_workspace({"MEDIA_WORKSPACE_GC": value}, "/b/run-x") is True


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "keep"])
def test_gc_falsy_values_disable(value):
    assert should_gc_workspace({"MEDIA_WORKSPACE_GC": value}, "/b/run-x") is False


def test_gc_never_touches_session_keyed_dir():
    """A pinned session key => the dir is the session's, not this run's; the
    loop must never reclaim it even if GC is requested."""
    env = {"MEDIA_WORKSPACE_GC": "reclaim", "MEDIA_WORKSPACE_KEY": "session-7"}
    assert should_gc_workspace(env, "/b/session-7") is False


def test_gc_none_env_is_false():
    assert should_gc_workspace(None, "/b/run-x") is False


def test_workspace_gc_command_is_quoted_rm():
    assert workspace_gc_command("/tmp/base/run-x") == "rm -rf /tmp/base/run-x"
    # paths with spaces are quoted so the shell sees a single argument
    assert workspace_gc_command("/tmp/a b/run x") == "rm -rf '/tmp/a b/run x'"
