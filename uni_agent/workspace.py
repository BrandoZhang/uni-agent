"""Per-run / per-session media workspace isolation.

Pure, dependency-free helpers (no verl / numpy import chain) so they can be
unit-tested in isolation and reused outside the agent loop. ``UniAgentLoop``
calls these once per rollout to derive where a run's artifacts + cost ledger
live, keyed by a ``workspace_key``.

Why this exists: on a **shared** filesystem (``local_native`` / ``host``, where
every rollout sees the same host FS) the agent picks its own output names
(``ref.png``, ``final.mp4``, ...), so two concurrent tasks that pick the same
name would overwrite each other's artifacts and share the cost ledger.
Container backends (``modal`` / ``vefaas`` / ``local``) isolate the FS per run
for free, so this derivation is a harmless no-op there.
"""

from __future__ import annotations

import shlex
from pathlib import Path


def resolve_media_workspace(env_variables: dict[str, str] | None, run_id: str) -> tuple[dict[str, str], str | None]:
    """Derive a per-run / per-session media workspace, keyed by a ``workspace_key``.

    Isolation is **opt-in** via ``MEDIA_WORKSPACE_BASE``: when present, the
    effective workspace is ``<base>/<workspace_key>``, where

    - ``workspace_key`` = ``MEDIA_WORKSPACE_KEY`` if the caller pinned a stable
      session id (an interactive, *resumable* task that spans many model
      invocations — the same key must map to the same dir so prior artifacts
      persist), else the per-rollout ``run_id`` (a one-shot training/eval
      trajectory — a fresh uuid per run is exactly right).

    ``MEDIA_WORKSPACE`` (the reward's film-probe dir) is always set to that
    dir; ``MEDIA_USAGE_LOG`` (the cost ledger) is set under it *unless* the
    caller pinned one. Without a base the env is returned unchanged
    (back-compat: an explicit static ``MEDIA_WORKSPACE`` is respected as-is).

    Returns ``(updated_env_variables, workspace_path or None)``. ``None`` means
    isolation was not requested and nothing was derived.
    """
    env = dict(env_variables or {})
    base = env.get("MEDIA_WORKSPACE_BASE")
    if not base:
        return env, None
    workspace_key = env.get("MEDIA_WORKSPACE_KEY") or run_id
    workspace = str(Path(base).expanduser() / workspace_key)
    env["MEDIA_WORKSPACE"] = workspace
    env.setdefault("MEDIA_USAGE_LOG", str(Path(workspace) / "usage.jsonl"))
    return env, workspace


def compose_post_setup_cmd(existing: str | None, workspace: str) -> str:
    """Ensure the sandbox shell starts inside ``workspace`` (creating it),
    preserving any pre-existing ``post_setup_cmd``."""
    cd = f"mkdir -p {shlex.quote(workspace)} && cd {shlex.quote(workspace)}"
    return f"{existing} && {cd}" if existing else cd
