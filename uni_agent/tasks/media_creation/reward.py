"""Cost-aware reward for the media-creation task.

    reward = quality_proxy - cost_weight * total_tokens   (clamped to [-1, 1])

Two properties matter for training signal correctness:

* **Cost comes from THIS episode's transcript** -- every ``media-ai`` result the
  agent produced carries a ``usage`` block on stdout; we sum its ``total_tokens``
  with no hardcoded tool allowlist. The shared usage ledger is deliberately never
  read (it would collide across parallel rollouts).
* **The final film is discovered from the transcript** -- the last ``concat``
  output (media-ai ``operation == "video.concat"``), else the last produced clip
  -- then checked to actually exist (non-empty) in the sandbox. So the score
  doesn't depend on the agent naming the file a fixed way.

``quality_proxy`` is a placeholder (did we produce a real film with enough
footage) -- swap in a VLM / aesthetic judge for production.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _parse_media_json(observation: str) -> dict | None:
    """Extract the JSON a ``media-ai`` command printed from a tool observation.

    The shell tool wraps stdout as ``[exit code: N]\\n[stdout]\\n<...>``; media-ai
    prints exactly one JSON object. Scan for the last ``{...}`` line and parse it.
    """
    if not observation:
        return None
    for line in reversed(observation.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            return obj if isinstance(obj, dict) else None
    return None


def cost_and_film_from_transcript(transcript: list[dict[str, Any]]) -> tuple[dict, str | None]:
    """Sum generation cost and find the final film from the episode's tool messages.

    Generic accounting: any tool message carrying a media-ai ``usage`` block
    contributes its ``total_tokens`` (async jobs are deduped by task id, so a
    re-polled ``job query`` isn't double-counted). Footage is counted by result
    shape (``meta.seconds``), not tokens, so a zero-token real clip still counts.
    """
    totals = {"calls": 0, "images_generated": 0, "video_seconds": 0, "total_tokens": 0}
    final_film: str | None = None
    last_video: str | None = None
    seen_task_ids: set[str] = set()

    for message in transcript:
        if message.get("role") != "tool":
            continue
        payload = _parse_media_json(message.get("content") or "")
        if payload is None or not payload.get("ok", True):
            continue

        kind = payload.get("kind")
        # Final film = last concat output; detect by media-ai operation (name-
        # independent, since generation flows through the generic shell tool).
        if payload.get("operation") == "video.concat" and payload.get("path"):
            final_film = payload["path"]
        elif kind == "video" and payload.get("path"):
            last_video = payload["path"]

        usage = payload.get("usage") or {}
        meta = payload.get("meta") or {}
        tokens = int(usage.get("total_tokens", 0) or 0)
        images = int(usage.get("generated_images", 0) or 0)
        seconds = int(meta.get("seconds", 0) or 0) if kind == "video" else 0

        # Skip a repeated poll of an already-counted async task (finalized job
        # JSON carries a top-level id == the task id).
        task_id = payload.get("id")
        duplicate = bool(task_id) and task_id in seen_task_ids
        if not duplicate and (tokens or images or seconds):
            totals["calls"] += 1
            totals["total_tokens"] += tokens
            totals["images_generated"] += images
            totals["video_seconds"] += seconds
            if task_id:
                seen_task_ids.add(task_id)

    return totals, (final_film or last_video)


async def _film_exists(sandbox, path: str | None) -> tuple[bool, int]:
    """Whether ``path`` is a non-empty file in the sandbox, and its byte size."""
    if not path:
        return False, 0
    res = await sandbox.exec(["stat", "-c", "%s", path])
    if res.exit_code != 0:
        return False, 0
    try:
        size = int((res.stdout or "0").strip())
    except ValueError:
        return False, 0
    return size > 0, size


def _quality_proxy(film_exists: bool, totals: dict, min_seconds: int) -> float:
    """Placeholder quality: 1.0 for a real film with enough footage, 0.6 for a
    shorter one, 0.0 for none. Swap in a VLM/aesthetic judge for production."""
    if not film_exists:
        return 0.0
    return 1.0 if totals.get("video_seconds", 0) >= min_seconds else 0.6


async def compute_reward(
    metadata: dict,
    sandbox,
    agent_result,
    *,
    cost_weight: float = 1e-4,
    min_seconds: int = 4,
) -> dict:
    """Score one media-creation episode. Returns a JSON-able info dict incl. ``reward``."""
    transcript = getattr(agent_result, "transcript", None) or []
    totals, final_film = cost_and_film_from_transcript(transcript)
    film_exists, film_bytes = await _film_exists(sandbox, final_film)

    quality = _quality_proxy(film_exists, totals, min_seconds)
    total_tokens = int(totals["total_tokens"])
    raw = quality - cost_weight * total_tokens
    reward = max(-1.0, min(1.0, raw))

    info = {
        "reward": reward,
        "quality_proxy": quality,
        "total_tokens": total_tokens,
        "cost_penalty": cost_weight * total_tokens,
        "video_seconds": totals["video_seconds"],
        "images_generated": totals["images_generated"],
        "generation_calls": totals["calls"],
        "final_film": final_film,
        "final_film_exists": film_exists,
        "final_film_bytes": film_bytes,
        # The agent's own LLM token usage (separate from generation cost above).
        "llm_tokens": int(getattr(agent_result, "info", {}).get("total_tokens", 0) or 0),
        "exit_reason": getattr(agent_result, "info", {}).get("exit_reason"),
    }
    logger.info(
        f"reward={reward:.4f} quality={quality:.2f} gen_tokens={total_tokens} "
        f"seconds={totals['video_seconds']} film={final_film} exists={film_exists}"
    )
    return info


__all__ = ["compute_reward", "cost_and_film_from_transcript"]
