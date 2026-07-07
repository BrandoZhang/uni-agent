"""Reward spec for the multimodal creation agent.

Demonstrates **cost as an evaluation metric**: the score rewards actually
producing the film and penalizes the token cost spent generating it, so a
policy that meets the brief with fewer / cheaper generations scores higher.

    reward = quality_proxy - cost_weight * total_tokens   (clamped to [-1, 1])

- ``quality_proxy`` (in [0, 1]): did we produce a non-empty final film, and
  did it use at least a couple of shots. This is a *placeholder* for a real
  quality signal (VLM-as-judge / aesthetic / preference model) -- swap it in
  for a production reward.
- ``total_tokens``: summed from the usage ledger the media tools write.

Config keys (all optional; env vars fill the gaps):
    workspace    -> directory holding final.mp4 + usage.jsonl (or $MEDIA_WORKSPACE)
    usage_log    -> ledger path (or $MEDIA_USAGE_LOG, or <workspace>/usage.jsonl)
    final_film   -> path to the film (or <workspace>/final.mp4)
    cost_weight  -> penalty per token (default 1e-4)

This reward reads host-side files, matching the ``local_native`` demo; for a
container runtime, point the paths at a shared/bind-mounted location.
"""

from __future__ import annotations

import os
from pathlib import Path

from uni_agent.async_logging import get_logger
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.tools.media_gen import mediakit
from uni_agent.utils import auto_await


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

    def _quality_proxy(self, totals: dict) -> float:
        """Placeholder quality signal. Replace with a VLM/aesthetic judge."""
        if not (self.final_film.is_file() and self.final_film.stat().st_size > 0):
            return 0.0
        # a short film should be at least a couple of shots' worth of footage
        return 1.0 if totals.get("video_seconds", 0) >= 4 else 0.6

    @auto_await
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple[float, dict]:
        totals = mediakit.summarize_usage(self.usage_log)
        total_tokens = int(totals.get("total_tokens", 0))
        quality = self._quality_proxy(totals)
        raw = quality - self.cost_weight * total_tokens
        score = max(-1.0, min(1.0, raw))

        info = {
            "score": score,
            "quality_proxy": quality,
            "total_tokens": total_tokens,
            "cost_weight": self.cost_weight,
            "cost_penalty": self.cost_weight * total_tokens,
            "final_film_exists": self.final_film.is_file(),
            "usage_totals": totals,
        }
        self.logger.info(
            f"quality={quality:.2f} tokens={total_tokens} "
            f"penalty={self.cost_weight * total_tokens:.3f} -> reward={score:.3f}"
        )
        return score, info
