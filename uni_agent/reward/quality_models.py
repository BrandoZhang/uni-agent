"""Pluggable image-quality / alignment reward models for the image agent.

The `image_agent` reward's *quality* slot is where a real perceptual/preference model goes
-- the DanceGRPO "reward model on generated media" route and the DeepEyes "LLM/VLM-as-judge"
route. A scorer maps ``(image_path, prompt) -> float`` in ``[0, 1]`` (higher = better).

Selected via the reward config ``quality_model`` (a name, or ``{"name": ..., **kwargs}``):

  - ``"vlm_judge"``    -- a vision LLM judge over an OpenAI-compatible endpoint. Portable
                          (any served VLM), no extra weights; mirrors DeepEyes' judge and
                          scores both quality AND prompt alignment. **Fully implemented here.**
  - ``"image_reward"`` -- ImageReward (``pip install image-reward``), a text-image human
                          preference model. Score is min-max/sigmoid-normalized to ``[0, 1]``.
  - ``"hpsv3"``        -- HPSv3 (``pip install hpsv3`` + weights), a human-preference score.

**These are meaningful only on REAL generated images.** media-ai's ``mock`` provider emits a
placeholder card (a colored rectangle with the prompt text drawn on it), so any quality model
scores noise there -- keep the quality slot OFF for the offline mock demo (it is off by
default) and turn it on only with a real ``MEDIA_PROVIDER``.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class QualityScorer(Protocol):
    def score(self, image_path: str, prompt: str) -> float:  # pragma: no cover - protocol
        """Return a quality/alignment score in [0, 1] for one generated image."""
        ...


def _image_data_uri(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class VLMJudgeScorer:
    """Score an image with a vision LLM over an OpenAI-compatible endpoint.

    Asks the judge to rate, on a 1-10 scale, how well the image matches the prompt and how
    good its overall quality is; the integer is normalized to ``[0, 1]``. This is the most
    portable quality reward (no extra model weights) and is the DeepEyes judge pattern applied
    to a generated image. ``client`` may be injected for testing; otherwise an ``openai.OpenAI``
    client is built from ``base_url``/``api_key``.
    """

    _PROMPT = (
        "You are a strict image-quality judge. You are given a text prompt and an image that "
        "was generated from it. Rate, on an INTEGER scale from 1 (terrible) to 10 (excellent), "
        "how well the image matches the prompt AND how good its visual quality is. "
        "Reply with ONLY the integer.\n\nPrompt: {prompt}"
    )

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "EMPTY",
        client: Any | None = None,
        timeout: float = 60.0,
    ):
        import os

        self.model = model or os.environ.get("JUDGE_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=base_url or os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
                api_key=api_key,
                timeout=timeout,
            )

    def score(self, image_path: str, prompt: str) -> float:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._PROMPT.format(prompt=prompt)},
                        {"type": "image_url", "image_url": {"url": _image_data_uri(image_path)}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=8,
        )
        text = (completion.choices[0].message.content or "").strip()
        match = re.search(r"\d+", text)
        if not match:
            logger.warning("vlm_judge: could not parse a rating from %r; scoring 0.0", text)
            return 0.0
        rating = max(1, min(10, int(match.group(0))))
        return (rating - 1) / 9.0  # 1 -> 0.0, 10 -> 1.0


class ImageRewardScorer:
    """ImageReward (https://github.com/THUDM/ImageReward) preference model.

    ``pip install image-reward``. Scores are roughly in ``[-2, 2]``; squashed to ``[0, 1]``
    with a logistic. The model is loaded lazily and cached on the instance.
    """

    def __init__(self, model_name: str = "ImageReward-v1.0", device: str | None = None):
        try:
            import ImageReward as RM
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "quality_model='image_reward' requires the `image-reward` package (`pip install image-reward`)."
            ) from e
        self._model = RM.load(model_name) if device is None else RM.load(model_name, device=device)

    def score(self, image_path: str, prompt: str) -> float:
        import math

        raw = float(self._model.score(prompt, [image_path]))
        return _clip01(1.0 / (1.0 + math.exp(-raw)))  # logistic squash to [0, 1]


class HPSv3Scorer:
    """HPSv3 (https://github.com/MizzenAI/HPSv3) human-preference score.

    ``pip install hpsv3`` (+ model weights). HPSv3 returns a preference logit/μ; we squash it
    to ``[0, 1]`` with a logistic. The exact call is wrapped defensively because HPSv3's Python
    API has shifted across releases -- verify ``infer(...)`` against your installed version and
    adjust ``_raw_score`` if needed.
    """

    def __init__(self, **kwargs):
        try:
            from hpsv3 import HPSv3RewardInferencer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "quality_model='hpsv3' requires the `hpsv3` package (`pip install hpsv3`) and its weights."
            ) from e
        self._inferencer = HPSv3RewardInferencer(**kwargs)

    def _raw_score(self, image_path: str, prompt: str) -> float:
        # HPSv3RewardInferencer.reward(images, prompts) -> list of (mu, sigma) tensors in recent
        # releases. Take mu of the first result.
        out = self._inferencer.reward([image_path], [prompt])
        first = out[0]
        if hasattr(first, "__len__") and not isinstance(first, (int, float)):
            first = first[0]
        return float(first)

    def score(self, image_path: str, prompt: str) -> float:
        import math

        return _clip01(1.0 / (1.0 + math.exp(-self._raw_score(image_path, prompt))))


_SCORERS: dict[str, type] = {
    "vlm_judge": VLMJudgeScorer,
    "image_reward": ImageRewardScorer,
    "hpsv3": HPSv3Scorer,
}

# Cache scorers by config so repeated per-sample reward construction reuses one instance
# (loading an ImageReward/HPSv3 torch model per rollout would be prohibitively expensive).
_SCORER_CACHE: dict[str, QualityScorer] = {}


def get_quality_scorer(config: str | dict | None) -> QualityScorer | None:
    """Build (and cache) a quality scorer from ``reward.quality_model`` config.

    ``None`` / falsy -> no scorer (the reward falls back to the offline keyword proxy).
    A string -> scorer name with defaults. A dict -> ``{"name": ..., **ctor_kwargs}``.
    """
    if not config:
        return None
    if isinstance(config, str):
        name, kwargs = config, {}
    elif isinstance(config, dict):
        cfg = dict(config)
        name = cfg.pop("name", None) or cfg.pop("type", None)
        kwargs = cfg
    else:
        raise TypeError(f"quality_model must be str | dict | None, got {type(config)!r}")
    if name not in _SCORERS:
        raise ValueError(f"Unknown quality_model {name!r}. Available: {sorted(_SCORERS)}")

    cache_key = repr((name, sorted(kwargs.items(), key=lambda kv: kv[0])))
    if cache_key not in _SCORER_CACHE:
        _SCORER_CACHE[cache_key] = _SCORERS[name](**kwargs)
    return _SCORER_CACHE[cache_key]
