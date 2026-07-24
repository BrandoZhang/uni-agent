"""Tests for the media_creation task: registration, config, prompt, and run().

``run()`` is exercised hermetically with a fake sandbox + fake agent (no real
sandbox, no LLM, no media-ai) so it stays a light offline test; the full stack is
covered end-to-end by ``examples/media_creation/demo.py``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from uni_agent.agents import AgentResult
from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks.media_creation.task import MediaCreationTask, MediaCreationTaskConfig
from uni_agent.tasks.registry import get_task_cls


def _tool_msg(payload: dict) -> dict:
    return {"role": "tool", "name": "shell",
            "content": f"Observation:\n[exit code: 0]\n[stdout]\n{json.dumps(payload)}"}


class _FakeSandbox:
    def __init__(self, files: dict[str, int] | None = None):
        self.files = files or {}
        self.scripts: list[str] = []
        self.uploads: list[tuple[str, str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def exec_shell(self, script, **kwargs):
        self.scripts.append(script)
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def exec(self, argv, **kwargs):
        path = argv[-1]
        if path in self.files:
            return ExecResult(exit_code=0, stdout=f"{self.files[path]}\n", stderr="")
        return ExecResult(exit_code=1, stdout="", stderr="missing")

    async def upload(self, local, remote):
        self.uploads.append((str(local), remote))


class _FakeAgent:
    def __init__(self, transcript):
        self._transcript = transcript
        self.messages = None

    async def run(self, *, sandbox, messages):
        self.messages = messages
        return AgentResult(transcript=self._transcript, info={"total_tokens": 42, "exit_reason": "finished"})


def _make_task(monkeypatch, cfg, sandbox, agent):
    task = MediaCreationTask(cfg)
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    monkeypatch.setattr(task, "build_agent", lambda: agent)
    return task


def test_task_is_registered():
    assert get_task_cls("media_creation") is MediaCreationTask


def test_config_defaults_and_name():
    cfg = MediaCreationTaskConfig(sandbox={"provider": "local"})
    assert cfg.name == "media_creation"
    assert cfg.cost_weight == 1e-4 and cfg.quality_min_seconds == 4
    assert cfg.workspace.startswith("/tmp/") and cfg.inject_skills is True


def test_run_builds_prompt_and_scores_reward(monkeypatch):
    cfg = MediaCreationTaskConfig(
        sandbox={"provider": "local"},
        workspace="/w",
        inject_skills=False,  # no media-ai needed for this hermetic test
        prompt=[{"role": "user", "content": "make a short film"}],
    )
    transcript = [
        _tool_msg({"ok": True, "kind": "video", "operation": "video.generate",
                   "path": "/w/s1.mp4", "usage": {"total_tokens": 1000}, "meta": {"seconds": 3}}),
        _tool_msg({"ok": True, "kind": "video", "operation": "video.generate",
                   "path": "/w/s2.mp4", "usage": {"total_tokens": 1000}, "meta": {"seconds": 3}}),
        _tool_msg({"ok": True, "kind": "video", "operation": "video.concat",
                   "path": "/w/final.mp4", "usage": {}, "meta": {"clips": 2}}),
    ]
    sandbox = _FakeSandbox(files={"/w/final.mp4": 500})
    agent = _FakeAgent(transcript)
    task = _make_task(monkeypatch, cfg, sandbox, agent)

    result = asyncio.run(task.run())

    # reward = quality(1.0, 6s >= 4s) - 1e-4 * 2000
    assert result.info["final_film"] == "/w/final.mp4" and result.info["final_film_exists"] is True
    assert result.info["total_tokens"] == 2000
    assert result.reward == 1.0 - 1e-4 * 2000
    assert result.accuracy == 1.0

    # the task built a system prompt (with the workspace) then appended the brief
    assert agent.messages[0]["role"] == "system" and "/w" in agent.messages[0]["content"]
    assert agent.messages[-1] == {"role": "user", "content": "make a short film"}
    # it also made the workspace
    assert any("mkdir -p /w" in s for s in sandbox.scripts)


def test_run_injects_skills_manifest_when_enabled(monkeypatch):
    pytest.importorskip("media_ai", reason="media-ai not installed")
    cfg = MediaCreationTaskConfig(sandbox={"provider": "local"}, workspace="/w", inject_skills=True,
                                  prompt=[{"role": "user", "content": "make art"}])
    sandbox = _FakeSandbox(files={})
    agent = _FakeAgent([])
    task = _make_task(monkeypatch, cfg, sandbox, agent)

    asyncio.run(task.run())

    assert "<available_skills>" in agent.messages[0]["content"]
    assert sandbox.uploads and sandbox.uploads[0][1] == cfg.skills_root


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
