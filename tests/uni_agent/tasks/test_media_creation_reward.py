"""Tests for the cost-aware media_creation reward.

Two properties that matter for training-signal correctness:

* cost is summed from THIS episode's transcript (the media-ai ``usage`` blocks in
  the shell tool observations), with no tool allowlist and async jobs deduped;
* the final film is discovered from the transcript (the last ``video.concat``
  output) and verified to exist in the sandbox.
"""

from __future__ import annotations

import asyncio
import json

from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks.media_creation.reward import compute_reward, cost_and_film_from_transcript


def _tool_msg(payload: dict) -> dict:
    """A ReAct tool message: the shell tool's observation wrapping media-ai's stdout."""
    stdout = json.dumps(payload)
    content = f"Observation:\n[exit code: 0]\n[stdout]\n{stdout}"
    return {"role": "tool", "name": "shell", "content": content}


def _img(path: str, tokens: int, images: int = 1) -> dict:
    return {"ok": True, "kind": "image", "operation": "image.generate", "path": path,
            "usage": {"total_tokens": tokens, "generated_images": images}}


def _vid(path: str, tokens: int, seconds: int) -> dict:
    return {"ok": True, "kind": "video", "operation": "video.generate", "path": path,
            "usage": {"total_tokens": tokens}, "meta": {"seconds": seconds}}


def _concat(path: str) -> dict:
    return {"ok": True, "kind": "video", "operation": "video.concat", "path": path,
            "usage": {}, "meta": {"clips": 2}}


class _AgentResult:
    def __init__(self, transcript, info=None):
        self.transcript = transcript
        self.info = info or {"total_tokens": 999, "exit_reason": "finished"}


class _FakeSandbox:
    """Minimal sandbox: ``exec(['stat','-c','%s',path])`` reports file sizes."""

    def __init__(self, files: dict[str, int] | None = None):
        self.files = files or {}

    async def exec(self, argv, **kwargs) -> ExecResult:
        path = argv[-1]
        if path in self.files:
            return ExecResult(exit_code=0, stdout=f"{self.files[path]}\n", stderr="")
        return ExecResult(exit_code=1, stdout="", stderr="No such file")


# ---- pure transcript accounting -------------------------------------------

def test_cost_and_film_summed_from_transcript():
    transcript = [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "step", "tool_calls": [{"id": "c1"}]},
        _tool_msg(_img("/w/ref.png", 100)),
        _tool_msg(_vid("/w/s1.mp4", 200, 3)),
        _tool_msg(_vid("/w/s2.mp4", 300, 3)),
        _tool_msg(_concat("/w/final.mp4")),
    ]
    totals, film = cost_and_film_from_transcript(transcript)
    assert totals["total_tokens"] == 600  # concat carries no tokens
    assert totals["calls"] == 3
    assert totals["images_generated"] == 1
    assert totals["video_seconds"] == 6
    assert film == "/w/final.mp4"  # detected by operation == video.concat


def test_film_falls_back_to_last_clip_without_concat():
    transcript = [_tool_msg(_vid("/w/only.mp4", 50, 2))]
    _, film = cost_and_film_from_transcript(transcript)
    assert film == "/w/only.mp4"


def test_zero_token_clip_still_counts_seconds():
    transcript = [_tool_msg({"ok": True, "kind": "video", "operation": "video.generate",
                             "path": "/w/s.mp4", "usage": {}, "meta": {"seconds": 5}})]
    totals, _ = cost_and_film_from_transcript(transcript)
    assert totals["video_seconds"] == 5 and totals["total_tokens"] == 0 and totals["calls"] == 1


def test_async_job_polled_twice_counts_once():
    finalized = {"ok": True, "kind": "video", "op": "query", "status": "succeeded", "id": "task-42",
                 "path": "/w/shot.mp4", "usage": {"total_tokens": 300}, "meta": {"seconds": 5}}
    totals, film = cost_and_film_from_transcript([_tool_msg(finalized), _tool_msg(finalized)])
    assert totals["total_tokens"] == 300 and totals["video_seconds"] == 5 and totals["calls"] == 1
    assert film == "/w/shot.mp4"


def test_non_json_and_failed_results_ignored():
    transcript = [
        {"role": "tool", "name": "shell", "content": "Observation:\n[exit code: 0]\n[stdout]\ntotal 0"},
        _tool_msg({"ok": False, "kind": "image", "path": "/w/x.png", "usage": {"total_tokens": 999}}),
        _tool_msg(_img("/w/ref.png", 42)),
    ]
    totals, _ = cost_and_film_from_transcript(transcript)
    assert totals["total_tokens"] == 42


# ---- end-to-end reward -----------------------------------------------------

def test_reward_rewards_film_and_penalizes_cost():
    transcript = [_tool_msg(_vid("/w/s1.mp4", 5000, 3)), _tool_msg(_vid("/w/s2.mp4", 5000, 3)),
                  _tool_msg(_concat("/w/final.mp4"))]
    sandbox = _FakeSandbox({"/w/final.mp4": 1000})
    info = asyncio.run(compute_reward({}, sandbox, _AgentResult(transcript), cost_weight=1e-4))
    assert info["final_film"] == "/w/final.mp4" and info["final_film_exists"] is True
    assert info["total_tokens"] == 10000 and info["quality_proxy"] == 1.0
    assert info["reward"] == 1.0 - 1e-4 * 10000  # 0.0


def test_reward_zero_quality_when_film_missing():
    transcript = [_tool_msg(_vid("/w/nope.mp4", 100, 2))]
    sandbox = _FakeSandbox({})  # nothing on disk
    info = asyncio.run(compute_reward({}, sandbox, _AgentResult(transcript), cost_weight=1e-4))
    assert info["final_film_exists"] is False and info["quality_proxy"] == 0.0
    assert info["reward"] == -1e-4 * 100


def test_reward_partial_quality_for_short_footage():
    transcript = [_tool_msg(_vid("/w/short.mp4", 100, 2))]  # 2s < min 4s
    sandbox = _FakeSandbox({"/w/short.mp4": 10})
    info = asyncio.run(compute_reward({}, sandbox, _AgentResult(transcript), cost_weight=1e-4, min_seconds=4))
    assert info["quality_proxy"] == 0.6


def test_reward_is_clamped():
    transcript = [_tool_msg(_vid("/w/nope.mp4", 10_000_000, 2))]
    info = asyncio.run(compute_reward({}, _FakeSandbox({}), _AgentResult(transcript), cost_weight=1.0))
    assert info["reward"] == -1.0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
