"""Offline unit tests for the image_agent composite reward."""

from __future__ import annotations

import json

import pytest

from uni_agent.reward.image_reward import DEFAULT_WEIGHTS, score_trajectory
from uni_agent.reward.registry import load_reward_spec

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover - environment without Pillow
    _HAS_PIL = False


def _make_png(path, size=(512, 512)) -> None:
    """Write a real PNG at ``size`` (PIL) or a minimal placeholder if PIL is absent."""
    if _HAS_PIL:
        Image.new("RGB", size, (200, 60, 60)).save(path)
    else:  # pragma: no cover
        # media-ai's stdlib fallback proves a PNG is writable without PIL; for the
        # test we only need a non-empty file to exercise the existence branch.
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


def _obs(image_path, size="512x512", prompt="a red bicycle at sunset") -> str:
    return "Observation:\n" + json.dumps(
        {"status": "ok", "image_path": str(image_path), "size": size, "prompt": prompt}
    )


def _gen_step(image_path, size="512x512", prompt="a red bicycle at sunset") -> dict:
    return {
        "exit_reason": "completed",
        "tool_results": [{"name": "generate_image", "status": "ok", "observation": _obs(image_path, size, prompt)}],
    }


def _finish_step() -> dict:
    return {
        "exit_reason": "finished",
        "tool_results": [{"name": "finish", "status": "ok", "observation": "done"}],
    }


def test_happy_path_full_credit(tmp_path):
    img = tmp_path / "out.png"
    _make_png(img, (512, 512))
    trajectory = [_gen_step(img, "512x512"), _finish_step()]

    score, info = score_trajectory(trajectory)

    assert info["tool_used"] is True
    assert info["reached_finish"] is True
    assert info["components"]["tool"] == 1.0
    assert info["components"]["format"] == 0.0
    if _HAS_PIL:
        assert info["components"]["artifact"] == 1.0
        assert info["artifact_detail"]["size_match"] is True
        # 0.8*1 + 0.2*0 + 1.2*1 + 0.5*0
        assert score == pytest.approx(DEFAULT_WEIGHTS["artifact"] + DEFAULT_WEIGHTS["tool"])
    else:  # pragma: no cover
        assert info["components"]["artifact"] == 1.0


def test_wrong_size_partial_artifact(tmp_path):
    if not _HAS_PIL:  # pragma: no cover
        pytest.skip("dimension check requires Pillow")
    img = tmp_path / "out.png"
    _make_png(img, (256, 256))  # actual 256x256 but agent requested 512x512
    trajectory = [_gen_step(img, "512x512"), _finish_step()]

    score, info = score_trajectory(trajectory)
    assert info["components"]["artifact"] == 0.5
    assert info["artifact_detail"]["size_match"] is False
    # tool credit requires a *valid* (full) artifact -> withheld here
    assert info["components"]["tool"] == 0.0


def test_no_tool_call_is_format_error():
    trajectory = [{"exit_reason": "format_error", "tool_results": []}]
    score, info = score_trajectory(trajectory)
    assert info["components"]["tool"] == 0.0
    assert info["components"]["artifact"] == 0.0
    assert info["components"]["format"] == -1.0
    assert score == pytest.approx(DEFAULT_WEIGHTS["format"] * -1.0)


def test_offline_quality_keywords(tmp_path):
    img = tmp_path / "out.png"
    _make_png(img, (512, 512))
    prompt = "a red bicycle at sunset, cinematic lighting"
    trajectory = [_gen_step(img, "512x512", prompt), _finish_step()]

    _, info = score_trajectory(trajectory, ground_truth={"keywords": ["bicycle", "sunset", "spaceship"]})
    # 2 of 3 keywords present, via the offline proxy
    assert info["components"]["quality"] == pytest.approx(2 / 3)
    assert info["quality_mode"] == "offline_keywords"


def test_real_quality_scorer_used(tmp_path):
    img = tmp_path / "out.png"
    _make_png(img, (512, 512))
    trajectory = [_gen_step(img, "512x512"), _finish_step()]

    class _FakeScorer:
        def score(self, image_path, prompt):
            assert image_path.endswith(".png") and prompt
            return 0.9

    _, info = score_trajectory(trajectory, quality_scorer=_FakeScorer())
    assert info["components"]["quality"] == pytest.approx(0.9)
    assert info["quality_mode"].startswith("reward_model:")


def test_quality_scorer_failure_falls_back(tmp_path):
    img = tmp_path / "out.png"
    _make_png(img, (512, 512))
    trajectory = [_gen_step(img, "512x512", "a red bicycle at sunset"), _finish_step()]

    class _BoomScorer:
        def score(self, image_path, prompt):
            raise RuntimeError("no weights")

    _, info = score_trajectory(trajectory, ground_truth={"keywords": ["bicycle"]}, quality_scorer=_BoomScorer())
    # degrades gracefully to the offline proxy rather than losing the rollout
    assert info["quality_mode"] == "offline_keywords"
    assert info["components"]["quality"] == pytest.approx(1.0)


def test_registry_and_async_interface(tmp_path):
    img = tmp_path / "out.png"
    _make_png(img, (512, 512))
    trajectory = [_gen_step(img, "512x512"), _finish_step()]

    spec = load_reward_spec({"name": "image_agent", "run_id": "t", "ground_truth": {}})
    # @auto_await resolves the coroutine synchronously when no event loop is running.
    score, info = spec.compute_reward(interaction_result={"trajectory": trajectory})
    assert isinstance(score, float)
    assert "components" in info
