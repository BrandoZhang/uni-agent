"""Tests for the framework-neutral UI event translation."""

from __future__ import annotations

import json
from types import SimpleNamespace

from apps.creation_gui.events import cost_event, parse_tool_json, translate_step


def _tr(name, payload=None, status="ok", tool_call_id="c1"):
    obs = "Observation:\n" + json.dumps(payload) if payload is not None else ""
    return SimpleNamespace(name=name, status=status, tool_call_id=tool_call_id, action=f"{name} --run", observation=obs)


def _step(step_idx=1, thought="", tool_results=(), done=False, exit_reason=""):
    return SimpleNamespace(
        step_idx=step_idx,
        thought=thought,
        response="",
        tool_results=list(tool_results),
        done=done,
        exit_reason=exit_reason,
    )


# always-refuse artifact registrar (isolates translation from path confinement)
def _no_art(_path):
    return None


def test_parse_tool_json_last_line():
    assert parse_tool_json("noise\n" + json.dumps({"a": 1})) == {"a": 1}
    assert parse_tool_json("no json") is None


def test_cost_event_rolls_up_namespaced_metrics():
    metrics = {
        "cost/llm/prompt_tokens": 100,
        "cost/llm/cached_tokens": 40,
        "cost/llm/completion_tokens": 20,
        "cost/llm/total_tokens": 120,
        "cost/tool/total_tokens": 500,
        "cost/tool/text2image/total_tokens": 300,
        "cost/tool/text2video/total_tokens": 200,
    }
    ev = cost_event(metrics)
    assert ev["type"] == "cost"
    assert ev["llm"]["cached_tokens"] == 40
    assert ev["by_tool"] == {"text2image": 300, "text2video": 200}
    assert ev["tool_total_tokens"] == 500
    assert ev["total_tokens"] == 120 + 500  # llm total + tool total


def test_translate_emits_assistant_toolcall_result_cost():
    payload = {"ok": True, "kind": "image", "path": "/w/ref.png", "usage": {"total_tokens": 42}}
    step = _step(thought="planning the hero shot", tool_results=[_tr("text2image", payload)])
    events = translate_step(step, {"cost/llm/total_tokens": 10}, _no_art)
    types = [e["type"] for e in events]
    assert types == ["assistant", "tool_call", "tool_result", "cost"]
    assert events[0]["text"] == "planning the hero shot"
    assert events[1]["name"] == "text2image" and events[1]["command"] == "text2image --run"
    assert events[2]["status"] == "ok" and events[2]["usage"] == {"total_tokens": 42}


def test_translate_inlines_artifact_when_registrable():
    payload = {"ok": True, "kind": "video", "path": "/w/shot1.mp4", "usage": {"total_tokens": 99}}
    step = _step(tool_results=[_tr("image2video", payload)])
    events = translate_step(step, {}, lambda p: "/api/artifacts/abc123")
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["artifact"] == {"url": "/api/artifacts/abc123", "kind": "video"}


def test_translate_omits_artifact_when_registrar_refuses():
    payload = {"ok": True, "kind": "image", "path": "/etc/passwd"}
    step = _step(tool_results=[_tr("text2image", payload)])
    events = translate_step(step, {}, _no_art)  # registrar returns None -> no artifact
    result = next(e for e in events if e["type"] == "tool_result")
    assert "artifact" not in result


def test_translate_non_media_result_has_no_artifact():
    payload = {"ok": True, "totals": {"total_tokens": 5}}  # media_usage-style, no kind/path
    step = _step(tool_results=[_tr("media_usage", payload)])
    events = translate_step(step, {}, lambda p: "/api/artifacts/x")
    result = next(e for e in events if e["type"] == "tool_result")
    assert "artifact" not in result


def test_translate_done_event():
    step = _step(tool_results=[_tr("finish", None)], done=True, exit_reason="finished")
    events = translate_step(step, {}, _no_art)
    assert events[-1] == {"type": "done", "step": 1, "exit_reason": "finished"}


def test_translate_skips_empty_assistant_text():
    step = _step(thought="", tool_results=[_tr("text2image", {"kind": "image", "path": "/w/a.png"})])
    events = translate_step(step, {}, _no_art)
    assert all(e["type"] != "assistant" for e in events)
