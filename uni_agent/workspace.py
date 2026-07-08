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


# Values that turn run-end workspace GC on (case-insensitive). GC is off unless
# explicitly requested, so demo/eval runs keep their artifacts to inspect and
# only large-scale/training runs reclaim disk.
_GC_TRUTHY = frozenset({"1", "true", "yes", "on", "reclaim"})


def should_gc_workspace(env_variables: dict[str, str] | None, media_workspace: str | None) -> bool:
    """Whether ``UniAgentLoop`` should reclaim the per-run workspace at run end.

    True only when **all** hold:

    - a per-run workspace was actually derived (``media_workspace`` is not None);
    - the caller did **not** pin a ``MEDIA_WORKSPACE_KEY`` — a session-keyed dir
      is owned by the (resumable) session, not this run, so the loop must never
      delete it;
    - ``MEDIA_WORKSPACE_GC`` opts in (default off).

    The deletion itself must run **through the env** (``rm -rf`` inside the
    sandbox), not a host-side ``shutil.rmtree``: the artifacts live on the
    container's FS for container backends and on the host FS for
    ``local_native`` / ``host``. Going through the env deletes whichever one is
    real and never risks a host path for a container run. On container backends
    it is a harmless no-op-ish redundancy (``env.close()`` reclaims the FS
    anyway); on a shared FS it is the only thing that reclaims disk.
    """
    if not media_workspace:
        return False
    env = env_variables or {}
    if env.get("MEDIA_WORKSPACE_KEY"):
        return False
    return str(env.get("MEDIA_WORKSPACE_GC", "")).strip().lower() in _GC_TRUTHY


def workspace_gc_command(workspace: str) -> str:
    """The (idempotent, best-effort) shell command that reclaims a run's
    workspace. Run it through ``env.communicate`` so it targets the FS the
    artifacts actually live on. ``cd /`` first so the shell isn't sitting in
    the directory being removed (avoids ``getcwd`` errors on later commands)."""
    return f"cd / && rm -rf {shlex.quote(workspace)}"


def gc_would_delete(workspace: str, protected: str) -> bool:
    """True if ``rm -rf <workspace>`` would also remove ``protected`` — i.e.
    ``protected`` is ``workspace`` itself or lives inside it.

    Guards against a misconfiguration where ``MEDIA_WORKSPACE_BASE`` is pointed
    at (a parent of) the loop's ``log_dir``: then ``<base>/<run_id>`` coincides
    with ``<log_dir>/<run_id>`` = the run's ``output_dir``, and reclaiming the
    workspace would silently destroy the just-saved training record
    (rollout_cache.pkl / interaction_result.json / run.log).
    """
    ws = Path(workspace).expanduser()
    pr = Path(protected).expanduser()
    try:
        pr.resolve(strict=False).relative_to(ws.resolve(strict=False))
        return True
    except ValueError:
        return False
