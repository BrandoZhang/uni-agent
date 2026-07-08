"""Framework-neutral UI event translation for the creation GUI.

Pure functions only — **no uni-agent import**. Given a completed interaction
*step* (any object exposing ``.step_idx`` / ``.thought`` / ``.response`` /
``.tool_results`` / ``.done`` / ``.exit_reason``) plus the live cost-metrics
dict, produce a list of plain-dict UI events. This is the contract the web GUI
consumes over SSE; keeping it pure means it is trivially unit-testable and the
GUI never depends on uni-agent.

Event shapes (``type`` discriminates):
    {type: "assistant",   step, text}
    {type: "tool_call",   step, id, name, command}
    {type: "tool_result", step, id, name, status, kind?, usage?, artifact?:{url,kind}}
    {type: "cost",        llm:{...}, by_tool:{...}, tool_total_tokens, total_tokens}
    {type: "done",        step, exit_reason}
"""

from __future__ import annotations

import json

_MEDIA_KINDS = frozenset({"image", "video"})
_LLM_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "total_tokens")


def parse_tool_json(observation: str) -> dict | None:
    """Extract the JSON result a media CLI printed (the last ``{...}`` line)."""
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


def cost_event(metrics: dict) -> dict:
    """Roll the ``cost/*`` counters in ``metrics`` into a UI cost event."""
    llm = {f: int(metrics.get(f"cost/llm/{f}", 0) or 0) for f in _LLM_FIELDS}
    by_tool: dict[str, int] = {}
    for key, value in metrics.items():
        if key.startswith("cost/tool/") and key.endswith("/total_tokens") and key != "cost/tool/total_tokens":
            name = key[len("cost/tool/") : -len("/total_tokens")]
            by_tool[name] = int(value or 0)
    tool_total = int(metrics.get("cost/tool/total_tokens", 0) or 0)
    return {
        "type": "cost",
        "llm": llm,
        "by_tool": by_tool,
        "tool_total_tokens": tool_total,
        "total_tokens": llm["total_tokens"] + tool_total,
    }


def translate_step(step, metrics: dict, register_artifact) -> list[dict]:
    """Turn one completed step + live metrics into an ordered list of UI events.

    ``register_artifact(path) -> url | None`` maps a local artifact path to a
    servable URL (returning ``None`` when the path is outside the served root),
    so the GUI can inline images/videos without ever seeing a filesystem path.
    """
    events: list[dict] = []
    idx = getattr(step, "step_idx", None)

    text = (getattr(step, "thought", "") or getattr(step, "response", "") or "").strip()
    if text:
        events.append({"type": "assistant", "step": idx, "text": text})

    for tr in getattr(step, "tool_results", []) or []:
        name = getattr(tr, "name", "")
        tid = getattr(tr, "tool_call_id", "")
        events.append({"type": "tool_call", "step": idx, "id": tid, "name": name, "command": getattr(tr, "action", "")})

        result: dict = {
            "type": "tool_result",
            "step": idx,
            "id": tid,
            "name": name,
            "status": getattr(tr, "status", ""),
        }
        payload = parse_tool_json(getattr(tr, "observation", "") or "")
        if payload:
            kind = payload.get("kind")
            result["kind"] = kind
            if isinstance(payload.get("usage"), dict):
                result["usage"] = payload["usage"]
            path = payload.get("path")
            if kind in _MEDIA_KINDS and path:
                url = register_artifact(path)
                if url:
                    result["artifact"] = {"url": url, "kind": kind}
        events.append(result)

    events.append(cost_event(metrics))

    if getattr(step, "done", False):
        events.append({"type": "done", "step": idx, "exit_reason": getattr(step, "exit_reason", "")})
    return events
