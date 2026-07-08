"""Tests for the cost-aware media_creation reward spec.

Verifies the two properties that matter for training signal correctness:

* **cost is derived from THIS run's trajectory** — any tool observation with a
  ``usage`` block contributes its ``total_tokens`` (no tool allowlist), and the
  shared usage ledger is deliberately never read (concurrency-safe);
* the **final film is discovered from the trajectory** (last ``concat_video``,
  else the last produced clip), so the score doesn't depend on a fixed filename.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from uni_agent.reward.media_creation import (
    MediaCreationRewardSpec,
    _cost_and_film_from_trajectory,
    _parse_tool_json,
)


def _tr(name: str, payload: dict | None, status: str = "ok"):
    """A fake tool_result: name + status + an ``Observation:``-wrapped JSON line
    (mirroring how AgentEnv wraps a media CLI's stdout)."""
    obs = ""
    if payload is not None:
        obs = "Observation:\n" + json.dumps(payload)
    return SimpleNamespace(name=name, status=status, observation=obs)


def _step(*tool_results):
    return SimpleNamespace(tool_results=list(tool_results))


def _img(path, tokens, images=1):
    return {"ok": True, "kind": "image", "path": path, "usage": {"total_tokens": tokens, "generated_images": images}}


def _vid(path, tokens, seconds):
    return {
        "ok": True,
        "kind": "video",
        "path": path,
        "usage": {"total_tokens": tokens, "completion_tokens": tokens},
        "meta": {"seconds": seconds},
    }


def _concat(path):
    return {"ok": True, "kind": "video", "path": path, "usage": {}}


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def test_parse_tool_json_from_wrapped_observation():
    payload = _img("/w/ref.png", 100)
    assert _parse_tool_json("Observation:\n" + json.dumps(payload)) == payload


def test_parse_tool_json_ignores_noise_and_takes_last_json():
    obs = "some log line\n" + json.dumps({"a": 1}) + "\nfinal: " + "\n" + json.dumps({"b": 2})
    assert _parse_tool_json(obs) == {"b": 2}


def test_parse_tool_json_none_on_no_json():
    assert _parse_tool_json("no json here") is None
    assert _parse_tool_json("") is None


# --------------------------------------------------------------------------
# cost + film extraction
# --------------------------------------------------------------------------


def test_cost_summed_across_tools():
    traj = [
        _step(_tr("text2image", _img("/w/ref.png", 100, images=1))),
        _step(_tr("image2video", _vid("/w/s1.mp4", 200, 3))),
        _step(_tr("text2video", _vid("/w/s2.mp4", 300, 2))),
    ]
    totals, film = _cost_and_film_from_trajectory(traj)
    assert totals["total_tokens"] == 600
    assert totals["calls"] == 3
    assert totals["images_generated"] == 1
    assert totals["video_seconds"] == 5
    assert totals["by_tool"] == {"text2image": 100, "image2video": 200, "text2video": 300}
    # no concat -> film is the last produced clip
    assert film == "/w/s2.mp4"


def test_concat_is_the_final_film():
    traj = [
        _step(_tr("image2video", _vid("/w/s1.mp4", 10, 2))),
        _step(_tr("concat_video", _concat("/w/final.mp4"))),
    ]
    _, film = _cost_and_film_from_trajectory(traj)
    assert film == "/w/final.mp4"


def test_last_concat_wins():
    traj = [
        _step(_tr("concat_video", _concat("/w/v1.mp4"))),
        _step(_tr("concat_video", _concat("/w/v2.mp4"))),
    ]
    _, film = _cost_and_film_from_trajectory(traj)
    assert film == "/w/v2.mp4"


def test_failed_tool_results_are_ignored():
    traj = [
        _step(_tr("text2image", _img("/w/ref.png", 999), status="error")),
        _step(_tr("text2video", _vid("/w/s.mp4", 50, 2))),
    ]
    totals, _ = _cost_and_film_from_trajectory(traj)
    assert totals["total_tokens"] == 50  # the errored 999 is not counted


def test_unknown_tool_with_usage_is_still_counted():
    # generic accounting: no hardcoded allowlist
    traj = [
        _step(_tr("some_future_tool", {"ok": True, "kind": "image", "path": "/w/x.png", "usage": {"total_tokens": 42}}))
    ]
    totals, _ = _cost_and_film_from_trajectory(traj)
    assert totals["total_tokens"] == 42


# --------------------------------------------------------------------------
# end-to-end reward
# --------------------------------------------------------------------------


def _spec(tmp_path, cost_weight=1e-4):
    # env=None -> host-side file probe (Path.stat); no sandbox needed in tests
    return MediaCreationRewardSpec(run_id="t", workspace=str(tmp_path), cost_weight=cost_weight, env=None)


def test_reward_rewards_film_and_penalizes_cost(tmp_path):
    film = tmp_path / "final.mp4"
    film.write_bytes(b"x" * 1000)  # non-empty
    traj = [
        _step(_tr("image2video", _vid(str(film), 5000, 3))),
        _step(_tr("text2video", _vid(str(tmp_path / "s2.mp4"), 5000, 3))),
        _step(_tr("concat_video", _concat(str(film)))),
    ]
    score, info = _spec(tmp_path).compute_reward({"trajectory": traj})
    assert info["final_film"] == str(film)
    assert info["final_film_exists"] is True
    assert info["total_tokens"] == 10000
    # quality 1.0 (film exists, >=4s footage) - 1e-4 * 10000 = 0.0
    assert info["quality_proxy"] == 1.0
    assert score == 1.0 - 1e-4 * 10000


def test_reward_zero_quality_when_no_film(tmp_path):
    missing = tmp_path / "nope.mp4"
    traj = [_step(_tr("text2video", _vid(str(missing), 100, 2)))]
    score, info = _spec(tmp_path).compute_reward({"trajectory": traj})
    assert info["final_film_exists"] is False
    assert info["quality_proxy"] == 0.0
    assert score == -1e-4 * 100  # negative: cost with no deliverable


def test_reward_partial_quality_for_short_footage(tmp_path):
    film = tmp_path / "short.mp4"
    film.write_bytes(b"x" * 10)
    traj = [_step(_tr("text2video", _vid(str(film), 100, 2)))]  # only 2s < 4s
    _, info = _spec(tmp_path).compute_reward({"trajectory": traj})
    assert info["quality_proxy"] == 0.6


def test_reward_is_clamped(tmp_path):
    missing = tmp_path / "nope.mp4"
    # enormous cost -> raw score below -1, must clamp
    traj = [_step(_tr("text2video", _vid(str(missing), 10_000_000, 2)))]
    score, _ = _spec(tmp_path, cost_weight=1.0).compute_reward({"trajectory": traj})
    assert score == -1.0


def test_reward_does_not_read_shared_ledger(tmp_path, monkeypatch):
    """Concurrency-safety: cost comes only from the trajectory, never a ledger
    file (a shared ledger would contaminate parallel rollouts)."""
    # plant a fat ledger that MUST be ignored
    ledger = tmp_path / "usage.jsonl"
    ledger.write_text(json.dumps({"tool": "text2image", "total_tokens": 999999}) + "\n")
    monkeypatch.setenv("MEDIA_USAGE_LOG", str(ledger))
    monkeypatch.setenv("MEDIA_WORKSPACE", str(tmp_path))
    traj = [_step(_tr("text2video", _vid(str(tmp_path / "nope.mp4"), 7, 2)))]
    _, info = _spec(tmp_path).compute_reward({"trajectory": traj})
    assert info["total_tokens"] == 7  # not 999999


def test_reward_empty_trajectory(tmp_path):
    score, info = _spec(tmp_path).compute_reward({"trajectory": []})
    assert info["total_tokens"] == 0
    assert info["final_film_exists"] is False
    assert score == 0.0
