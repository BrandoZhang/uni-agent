"""Reward spec for the multimodal creation agent.

Demonstrates **cost as an evaluation metric**: the score rewards actually
producing the film and penalizes the token cost spent generating it, so a
policy that meets the brief with fewer / cheaper generations scores higher.

    reward = quality_proxy - cost_weight * total_tokens   (clamped to [-1, 1])

- ``quality_proxy`` (in [0, 1]): did we produce a non-empty final film, and
  did it use at least a couple of shots. This is a *placeholder* for a real
  quality signal (VLM-as-judge / aesthetic / preference model) -- swap it in
  for a production reward.
- ``total_tokens``: **derived purely from this run's trajectory** -- any tool
  observation that carries a ``usage`` block contributes its ``total_tokens``
  (no hardcoded tool allowlist), summed from the run's own tool results. This
  is concurrency-safe (a shared usage-ledger file would collide across
  parallel rollouts, so it is deliberately NOT consulted) and mirrors how the
  ``search`` reward extracts its answer from the trajectory.

The final film is likewise **discovered from the trajectory** -- the last
``concat_video`` output, or the last produced video clip when there is no
concat (a valid one-shot deliverable) -- so the reward doesn't depend on the
agent naming the file a fixed way.

Config keys (all optional; env vars fill the gaps):
    workspace    -> fallback dir for final.mp4 (or $MEDIA_WORKSPACE)
    final_film   -> fallback film path (or <workspace>/final.mp4)
    cost_weight  -> penalty per token (default 1e-4)

File existence/size is probed through the ``AgentEnv`` abstraction (uniform
across a container sandbox and the local filesystem), falling back to a
host-side check when no env is available.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from uni_agent.async_logging import get_logger
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await


def _parse_tool_json(observation: str) -> dict | None:
    """Extract the JSON result a media CLI printed, from a tool observation.

    ``AgentEnv.run_action`` wraps stdout as ``Observation:\\n<stdout>``; the CLIs
    print a single JSON line. Tolerate the prefix and any surrounding text.
    """
    if not observation:
        return None
    for line in reversed(observation.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _cost_and_film_from_trajectory(trajectory: list) -> tuple[dict, str | None]:
    """Sum generation cost and find the final film from the run's own tool results.

    Cost is derived generically: **any** tool observation that carries a
    ``usage`` block contributes its ``total_tokens`` — no hardcoded tool
    allowlist — so new/renamed generation tools (or an async ``video_task``
    that finalizes a clip) are counted without special-casing. The film is the
    last ``concat_video`` output, falling back to the last produced video clip
    (a valid one-shot deliverable needs no concat).
    """
    totals = {"calls": 0, "images_generated": 0, "video_seconds": 0, "total_tokens": 0, "by_tool": {}}
    final_film: str | None = None
    last_video: str | None = None
    # Async video tasks can appear in the trajectory more than once (the agent
    # submits, then polls the same task id one or more times to download it).
    # Each poll re-reports the same usage, so we count a given task id once.
    seen_task_ids: set[str] = set()
    for step in trajectory:
        for tr in getattr(step, "tool_results", []) or []:
            name = getattr(tr, "name", "")
            if getattr(tr, "status", "") != "ok":
                continue
            payload = _parse_tool_json(getattr(tr, "observation", "") or "")
            if payload is None:
                continue
            kind = payload.get("kind")
            if name == "concat_video" and payload.get("path"):
                final_film = payload["path"]  # last concat wins
            elif kind == "video" and payload.get("path"):
                last_video = payload["path"]

            usage = payload.get("usage") or {}
            meta = payload.get("meta") or {}
            tok = int(usage.get("total_tokens", 0) or 0)
            imgs = int(usage.get("generated_images", 0) or 0)
            secs = int(meta.get("seconds", 0) or 0) if kind == "video" else 0
            # Identify a generation by its result shape, NOT by whether it
            # carried tokens: a real backend clip may report zero token usage,
            # and we still must count its footage. `concat_video` (a join, no
            # meta/usage) is naturally excluded — it has no tokens, images, or
            # seconds. Skip a repeated poll of an already-counted async task.
            task_id = payload.get("id") or meta.get("task_id")
            duplicate = bool(task_id) and task_id in seen_task_ids
            if not duplicate and (tok or imgs or secs):
                totals["calls"] += 1
                totals["total_tokens"] += tok
                totals["by_tool"][name] = totals["by_tool"].get(name, 0) + tok
                totals["images_generated"] += imgs
                totals["video_seconds"] += secs
                if task_id:
                    seen_task_ids.add(task_id)
    return totals, (final_film or last_video)


@register_reward_spec("media_creation")
class MediaCreationRewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        workspace: str | None = None,
        final_film: str | None = None,
        cost_weight: float = 1e-4,
        env=None,
        **kwargs,
    ):
        self.run_id = run_id
        self.workspace = Path(workspace or os.getenv("MEDIA_WORKSPACE", ".")).expanduser()
        self.final_film = Path(final_film or (self.workspace / "final.mp4")).expanduser()
        self.cost_weight = float(cost_weight)
        self.env = env
        self.logger = get_logger("media-creation-reward", run_id=run_id)

    async def _film_exists_and_size(self, path: str) -> tuple[bool, int]:
        """Check an artifact through the ``AgentEnv`` abstraction, which is
        uniform across a container sandbox and the local filesystem
        (``local_native``/``host``). Falls back to a host-side check when no
        env is available (e.g. standalone reward tests).

        Uses ``wc -c`` (POSIX, portable across GNU/BusyBox/macOS) rather than
        the GNU-only ``stat -c %s``.
        """
        if self.env is not None:
            try:
                q = shlex.quote(path)
                out = await self.env.communicate(
                    f"if [ -s {q} ]; then wc -c < {q}; else echo MISSING; fi", check="ignore"
                )
                for line in reversed((out or "").splitlines()):
                    line = line.strip()
                    if line.isdigit():
                        return True, int(line)
                    if line == "MISSING":
                        return False, 0
            except Exception as exc:  # noqa: BLE001 - fall back to host check
                self.logger.warning(f"env film probe failed ({exc}); falling back to host check")
        p = Path(path).expanduser()
        return (p.is_file() and p.stat().st_size > 0), (p.stat().st_size if p.is_file() else 0)

    def _quality_proxy(self, film_exists: bool, totals: dict) -> float:
        """Placeholder quality signal. Replace with a VLM/aesthetic judge."""
        if not film_exists:
            return 0.0
        # a short film should be at least a couple of shots' worth of footage
        return 1.0 if totals.get("video_seconds", 0) >= 4 else 0.6

    @auto_await
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple[float, dict]:
        # Cost is derived purely from THIS run's trajectory (concurrency-safe).
        # We deliberately do NOT fall back to the usage ledger: under parallel
        # rollouts the ledger is a shared file, so reading it would contaminate
        # one rollout's cost with others'. If the trajectory carries no usage,
        # the correct answer is 0 (no accountable generation), not the ledger.
        trajectory = interaction_result.get("trajectory", []) or []
        totals, film_from_traj = _cost_and_film_from_trajectory(trajectory)

        film = Path(film_from_traj).expanduser() if film_from_traj else self.final_film
        film_exists, film_size = await self._film_exists_and_size(str(film))
        total_tokens = int(totals.get("total_tokens", 0))
        quality = self._quality_proxy(film_exists, totals)
        raw = quality - self.cost_weight * total_tokens
        score = max(-1.0, min(1.0, raw))

        info = {
            "score": score,
            "quality_proxy": quality,
            "total_tokens": total_tokens,
            "cost_weight": self.cost_weight,
            "cost_penalty": self.cost_weight * total_tokens,
            "final_film": str(film),
            "final_film_exists": film_exists,
            "final_film_bytes": film_size,
            "usage_totals": totals,
        }
        self.logger.info(
            f"quality={quality:.2f} tokens={total_tokens} "
            f"penalty={self.cost_weight * total_tokens:.3f} -> reward={score:.3f}"
        )
        return score, info
