"""Reward spec for the multimodal creation agent.

Demonstrates **cost as an evaluation metric**: the score rewards actually
producing the film and penalizes the token cost spent generating it, so a
policy that meets the brief with fewer / cheaper generations scores higher.

    reward = quality_proxy - cost_weight * total_tokens   (clamped to [-1, 1])

- ``quality_proxy`` (in [0, 1]): did we produce a non-empty final film, and
  did it use at least a couple of shots. This is a *placeholder* for a real
  quality signal (VLM-as-judge / aesthetic / preference model) -- swap it in
  for a production reward.
- ``total_tokens``: **derived from the trajectory** -- each generation tool
  prints a JSON result with a ``usage`` block, so the cost is summed from the
  run's own tool observations. This is concurrency-safe (unlike a shared
  ledger file, which collides across parallel rollouts) and mirrors how
  ``search`` reward extracts its answer from the trajectory. The usage ledger
  is used only as a fallback when the trajectory has no usage.

The final film is likewise **discovered from the trajectory** (the last
``concat_video`` output), so the reward doesn't depend on the agent naming
the file a fixed way.

Config keys (all optional; env vars fill the gaps):
    workspace    -> fallback dir for final.mp4 / usage.jsonl (or $MEDIA_WORKSPACE)
    usage_log    -> fallback ledger path (or $MEDIA_USAGE_LOG)
    final_film   -> fallback film path (or <workspace>/final.mp4)
    cost_weight  -> penalty per token (default 1e-4)

File existence is checked host-side (matching the ``local_native`` demo); for
a container runtime, resolve the discovered path via ``env.read_file``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from media_ai import mediakit

from uni_agent.async_logging import get_logger
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

_GENERATION_TOOLS = {"text2image", "image2image", "text2video", "image2video", "ref2video"}


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
    """Sum generation cost and find the final film from the run's own tool results."""
    totals = {"calls": 0, "images_generated": 0, "video_seconds": 0, "total_tokens": 0, "by_tool": {}}
    final_film: str | None = None
    for step in trajectory:
        for tr in getattr(step, "tool_results", []) or []:
            name = getattr(tr, "name", "")
            if getattr(tr, "status", "") != "ok":
                continue
            payload = _parse_tool_json(getattr(tr, "observation", "") or "")
            if payload is None:
                continue
            if name == "concat_video" and payload.get("path"):
                final_film = payload["path"]  # last one wins
            if name in _GENERATION_TOOLS:
                usage = payload.get("usage") or {}
                tok = int(usage.get("total_tokens", 0) or 0)
                totals["calls"] += 1
                totals["total_tokens"] += tok
                totals["by_tool"][name] = totals["by_tool"].get(name, 0) + tok
                totals["images_generated"] += int(usage.get("generated_images", 0) or 0)
                meta = payload.get("meta") or {}
                if payload.get("kind") == "video":
                    totals["video_seconds"] += int(meta.get("seconds", 0) or 0)
    return totals, final_film


@register_reward_spec("media_creation")
class MediaCreationRewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        workspace: str | None = None,
        usage_log: str | None = None,
        final_film: str | None = None,
        cost_weight: float = 1e-4,
        env=None,
        **kwargs,
    ):
        self.run_id = run_id
        self.workspace = Path(workspace or os.getenv("MEDIA_WORKSPACE", ".")).expanduser()
        self.usage_log = Path(
            usage_log or os.getenv("MEDIA_USAGE_LOG", str(self.workspace / "usage.jsonl"))
        ).expanduser()
        self.final_film = Path(final_film or (self.workspace / "final.mp4")).expanduser()
        self.cost_weight = float(cost_weight)
        self.logger = get_logger("media-creation-reward", run_id=run_id)

    def _quality_proxy(self, film: Path, totals: dict) -> float:
        """Placeholder quality signal. Replace with a VLM/aesthetic judge."""
        if not (film.is_file() and film.stat().st_size > 0):
            return 0.0
        # a short film should be at least a couple of shots' worth of footage
        return 1.0 if totals.get("video_seconds", 0) >= 4 else 0.6

    @auto_await
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple[float, dict]:
        # Prefer the run's own trajectory (concurrency-safe); fall back to the ledger.
        trajectory = interaction_result.get("trajectory", []) or []
        totals, film_from_traj = _cost_and_film_from_trajectory(trajectory)
        source = "trajectory"
        if totals["total_tokens"] == 0:
            totals = mediakit.summarize_usage(self.usage_log)
            source = "ledger"

        film = Path(film_from_traj).expanduser() if film_from_traj else self.final_film
        total_tokens = int(totals.get("total_tokens", 0))
        quality = self._quality_proxy(film, totals)
        raw = quality - self.cost_weight * total_tokens
        score = max(-1.0, min(1.0, raw))

        info = {
            "score": score,
            "quality_proxy": quality,
            "total_tokens": total_tokens,
            "cost_weight": self.cost_weight,
            "cost_penalty": self.cost_weight * total_tokens,
            "final_film": str(film),
            "final_film_exists": film.is_file(),
            "cost_source": source,
            "usage_totals": totals,
        }
        self.logger.info(
            f"quality={quality:.2f} tokens={total_tokens} ({source}) "
            f"penalty={self.cost_weight * total_tokens:.3f} -> reward={score:.3f}"
        )
        return score, info
