"""Image-agent reward spec: a composite reward for a VLM that orchestrates image generation.

Structure mirrors DeepEyes' ``compute_score``
(``verl/recipe/deepeyes/deepeyes.py``: ``0.8*acc + 0.2*format + 1.2*tool``), adapted
for a *generation* agent and computable fully offline:

- **artifact** (the "acc" analog, weight 0.8): did the agent actually produce a valid
  image? File exists, is non-empty, and -- when Pillow is available -- its pixel
  dimensions match the size the agent requested.
- **format** (weight 0.2, applied as a penalty in ``{0, -1}``): did the rollout stay
  well-formed (no tool-call parse errors, and it reached ``finish``)?
- **tool** (weight 1.2): did the agent use ``generate_image`` *and* end up with a valid
  artifact -- i.e. it solved the task the intended way (DeepEyes gates ``tool`` on
  correctness the same way).
- **quality** (weight 0.5): perceptual/alignment quality of the final image.
  * With ``reward.quality_model`` set (``vlm_judge`` / ``image_reward`` / ``hpsv3`` -- see
    ``uni_agent/reward/quality_models.py``) this is a **real reward model** (the DanceGRPO
    reward-model route / DeepEyes VLM-judge). Needs real generated images + the model's deps;
    it is **meaningless on media-ai's ``mock`` provider** (placeholder cards) and is therefore
    OFF by default.
  * Otherwise it falls back to an offline keyword-overlap proxy against a ground-truth
    ``keywords`` list -- enough to make the offline demo/tests deterministic, not a real
    quality signal.

GRPO's group-relative baseline turns this absolute score into the "did this rollout do
better or worse than its siblings" signal, which is exactly the "变好还是变差" semantics
the design targets.
"""

import json
import logging
from pathlib import Path

from uni_agent.async_logging import get_logger
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.quality_models import get_quality_scorer
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"artifact": 0.8, "format": 0.2, "tool": 1.2, "quality": 0.5}


def _get(obj, key, default=None):
    """Read ``key`` from a StepOutput/ToolResult object *or* its dict form."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_observation(observation: str) -> dict | None:
    """Parse the outermost JSON object from a ``generate_image`` observation."""
    if not observation:
        return None
    start, end = observation.find("{"), observation.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(observation[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _last_generated_image(trajectory: list) -> dict | None:
    """Return the parsed observation of the last successful ``generate_image`` call."""
    for step in reversed(trajectory):
        for tr in reversed(_get(step, "tool_results", []) or []):
            if _get(tr, "name") == "generate_image" and _get(tr, "status") == "ok":
                obj = _parse_observation(_get(tr, "observation", "") or "")
                if obj and obj.get("status") == "ok":
                    return obj
    return None


def _artifact_score(image_info: dict | None) -> tuple[float, dict]:
    """Score the produced artifact: exists + non-empty (+ dims match request)."""
    detail: dict = {"exists": False, "requested_size": None, "actual_size": None, "size_match": None}
    if not image_info:
        return 0.0, detail

    path = image_info.get("image_path")
    detail["requested_size"] = image_info.get("size")
    if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
        return 0.0, detail
    detail["exists"] = True

    # Dimension check is a bonus when Pillow is available; otherwise "exists" is enough.
    try:
        from PIL import Image

        with Image.open(path) as im:
            actual = f"{im.size[0]}x{im.size[1]}"
        detail["actual_size"] = actual
        requested = (image_info.get("size") or "").lower().replace(" ", "")
        if requested and "x" in requested:
            match = actual == requested
            detail["size_match"] = match
            return (1.0 if match else 0.5), detail
    except Exception:
        # Pillow missing or unreadable -> existence alone earns full artifact credit.
        pass
    return 1.0, detail


def _keyword_quality(prompt: str, keywords: list[str]) -> float:
    """Offline quality proxy: fraction of ground-truth ``keywords`` present in the prompt.

    A deterministic stand-in used when no real ``quality_model`` is configured (e.g. the
    offline mock demo). Returns 0.0 when no keywords are supplied.
    """
    if keywords is None or len(keywords) == 0:
        return 0.0
    p = (prompt or "").lower()
    hits = sum(1 for kw in keywords if str(kw).lower() in p)
    return hits / len(keywords)


def _compute_quality(image_info, final_prompt, keywords, quality_scorer) -> tuple[float, str]:
    """Quality component: a real reward model if provided, else the offline keyword proxy.

    Returns ``(score_in_0_1, mode)``. A scorer that raises (missing weights, dead judge
    endpoint, ...) degrades gracefully to the keyword proxy so a rollout is never lost to a
    reward-side error.
    """
    path = (image_info or {}).get("image_path")
    if quality_scorer is not None and path and Path(path).is_file():
        if (image_info or {}).get("provider") == "mock":
            logger.warning(
                "quality_model is set but the image came from the 'mock' provider "
                "(placeholder card); the quality score is not meaningful."
            )
        try:
            q = quality_scorer.score(path, final_prompt)
            return max(0.0, min(1.0, float(q))), f"reward_model:{type(quality_scorer).__name__}"
        except Exception as e:
            logger.warning("quality_model scoring failed (%s); falling back to keyword proxy", e)
    return _keyword_quality(final_prompt, keywords), "offline_keywords"


def score_trajectory(
    trajectory: list,
    ground_truth: dict | None = None,
    weights: dict | None = None,
    quality_scorer=None,
) -> tuple[float, dict]:
    """Composite scorer. Offline/pure unless a ``quality_scorer`` is supplied.

    :param quality_scorer: optional object with ``score(image_path, prompt) -> [0,1]``
        (see ``uni_agent/reward/quality_models.py``). When ``None``, the quality component
        uses the offline keyword proxy, so the function stays pure/unit-testable.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    ground_truth = ground_truth or {}

    # --- tool usage & final prompt ---
    generate_calls = [
        tr
        for step in trajectory
        for tr in (_get(step, "tool_results", []) or [])
        if _get(tr, "name") == "generate_image"
    ]
    tool_used = any(_get(tr, "status") == "ok" for tr in generate_calls)

    image_info = _last_generated_image(trajectory)
    final_prompt = (image_info or {}).get("prompt", "")

    # --- components ---
    artifact, artifact_detail = _artifact_score(image_info)

    # tool credit only when the intended tool was used AND yielded a valid artifact.
    tool = 1.0 if (tool_used and artifact >= 1.0) else 0.0

    # format penalty: parse errors or never reaching a clean finish.
    exit_reasons = [_get(step, "exit_reason") for step in trajectory]
    is_format_error = ("format_error" in exit_reasons) or (not generate_calls)
    reached_finish = "finished" in exit_reasons
    if not reached_finish:
        is_format_error = True
    format_penalty = -1.0 if is_format_error else 0.0

    quality, quality_mode = _compute_quality(image_info, final_prompt, ground_truth.get("keywords", []), quality_scorer)

    final = (
        weights["artifact"] * artifact
        + weights["format"] * format_penalty
        + weights["tool"] * tool
        + weights["quality"] * quality
    )

    info = {
        "score": final,
        "components": {"artifact": artifact, "format": format_penalty, "tool": tool, "quality": quality},
        "quality_mode": quality_mode,
        "weights": weights,
        "tool_used": tool_used,
        "num_generate_calls": len(generate_calls),
        "final_prompt": final_prompt,
        "artifact_detail": artifact_detail,
        "reached_finish": reached_finish,
    }
    return final, info


@register_reward_spec("image_agent")
class ImageAgentRewardSpec(AbstractRewardSpec):
    """Composite reward for the image-generation agent (see module docstring)."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        ground_truth: dict | None = None,
        weights: dict | None = None,
        quality_model: str | dict | None = None,
        env=None,
        **kwargs,
    ):
        """:param quality_model: optional real quality reward -- a name or
        ``{"name": "vlm_judge"|"image_reward"|"hpsv3", ...}`` (see
        ``quality_models.get_quality_scorer``). ``None`` (default) uses the offline keyword
        proxy so the mock demo stays deterministic. Only turn it on with real images.
        """
        self.run_id = run_id
        self.ground_truth = ground_truth or {}
        self.weights = weights
        self.logger = get_logger("image-reward", run_id=run_id or "image-reward")
        try:
            self.quality_scorer = get_quality_scorer(quality_model)
        except Exception as e:
            # A misconfigured/optional quality model must not break training rollouts.
            self.logger.warning(f"quality_model {quality_model!r} unavailable ({e}); using keyword proxy")
            self.quality_scorer = None

    @auto_await
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple[float, dict]:
        trajectory = interaction_result.get("trajectory", []) or []
        score, info = score_trajectory(trajectory, self.ground_truth, self.weights, quality_scorer=self.quality_scorer)
        self.logger.info(
            f"image_agent reward={score:.3f} components={info['components']} "
            f"quality_mode={info['quality_mode']} tool_used={info['tool_used']} finish={info['reached_finish']}"
        )
        return score, info
