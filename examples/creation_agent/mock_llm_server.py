"""A minimal OpenAI-compatible chat-completions server that *scripts* a
storyboard, so the full ``AgentInteraction`` loop can be driven end-to-end
without a GPU or a real model.

There is no GPU in many sandboxes, and a real tool-calling model on CPU is
impractical for a demo. This server stands in for the LLM: on each
``POST /v1/chat/completions`` it looks at how many assistant turns have
already happened and returns the *next* scripted tool call (OpenAI
``tool_calls`` shape). That exercises the real loop: tool parsing, the bash
runtime, the media CLIs, the usage ledger, and the ``finish`` end-of-turn.

To run against a REAL model instead, ignore this file and point
``demo.py`` at any OpenAI-compatible tool-calling endpoint via ``BASE_URL``
(e.g. a vLLM server: ``vllm serve <model> --enable-auto-tool-choice
--tool-call-parser hermes``).

The script deliberately keeps 2 short shots at 480p to keep cost low.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _skill_location(messages: list[dict], tool: str = "image2video") -> str | None:
    """Pull a per-tool SKILL.md path out of the injected skills manifest.

    Demonstrates progressive disclosure: the 'model' reads the relevant
    tool's skill on demand. Defaults to the image2video skill (the
    consistency tool this scripted brief leans on).
    """
    for m in messages:
        if m.get("role") == "system":
            hit = re.search(rf"<location>([^<]*/{re.escape(tool)}/SKILL\.md)</location>", m.get("content") or "")
            if hit:
                return hit.group(1)
    return None


def build_script(workspace: str, skill_location: str | None) -> list[list[dict]]:
    """Ordered assistant turns; each turn is a LIST of tool calls the 'model'
    emits together (one turn can request several tools at once)."""
    ws = workspace.rstrip("/")
    read_skill = (
        {"name": "execute_bash", "arguments": {"command": f"cat {skill_location}"}}
        if skill_location
        else {"name": "execute_bash", "arguments": {"command": f"ls -la {ws}"}}
    )
    return [
        [read_skill],
        [
            {
                "name": "text2image",
                "arguments": {
                    "prompt": "A lone silver-suited astronaut on a red alien dune, cinematic teal-and-orange grade",
                    "output": f"{ws}/ref_hero.png",
                    "seed": 7,
                },
            }
        ],
        [
            # Both shots requested in ONE turn (two tool calls). The harness runs
            # each and returns both results — no bespoke batch tool needed.
            {
                "name": "image2video",
                "arguments": {
                    "first_frame": f"{ws}/ref_hero.png",
                    "prompt": "the astronaut slowly turns toward camera, gentle push-in",
                    "output": f"{ws}/shot1.mp4",
                    "seconds": 3,
                    "resolution": "480p",
                },
            },
            {
                "name": "text2video",
                "arguments": {
                    "prompt": "wide establishing shot of twin suns setting over the alien desert at dusk",
                    "output": f"{ws}/shot2.mp4",
                    "seconds": 3,
                    "resolution": "480p",
                },
            },
        ],
        [
            {
                "name": "concat_video",
                "arguments": {"inputs": [f"{ws}/shot1.mp4", f"{ws}/shot2.mp4"], "output": f"{ws}/final.mp4"},
            }
        ],
        [{"name": "media_usage", "arguments": {}}],
        [
            {
                "name": "finish",
                "arguments": {
                    "answer": (
                        "Done. Storyboard: (1) image2video of the hero astronaut from a locked "
                        f"reference frame, (2) text2video establishing shot. Final film: {ws}/final.mp4. "
                        "Cross-shot consistency came from ref_hero.png used as shot 1's first frame. "
                        "See the media_usage output above for the total token cost."
                    )
                },
            }
        ],
    ]


class _Handler(BaseHTTPRequestHandler):
    workspace = "/tmp/creation"

    def log_message(self, *args):  # silence default logging
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - health/models probes
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": "mock-director", "object": "model"}]})
        else:
            self._json(200, {"status": "ok"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid JSON"}})
            return

        messages = req.get("messages", [])
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        script = build_script(self.workspace, _skill_location(messages))
        step = script[n_assistant] if n_assistant < len(script) else script[-1]

        tool_calls = [
            {
                "id": f"call_{n_assistant}_{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
            }
            for i, call in enumerate(step)
        ]
        message = {
            "role": "assistant",
            "content": f"[director step {n_assistant}] calling {', '.join(c['name'] for c in step)}",
            "tool_calls": tool_calls,
        }
        # Synthesize a realistic usage breakdown so the fine-grained LLM cost
        # metrics (input / output / cached) are visible end-to-end. Rough
        # token estimate from the message payload; simulate prompt caching on
        # follow-up turns (a growing cached-input subset of the prompt).
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        for m in messages:
            for tc in m.get("tool_calls") or []:
                prompt_chars += len(str(tc.get("function", {}).get("arguments") or ""))
        prompt_tokens = prompt_chars // 4
        completion_tokens = 24
        cached_tokens = int(prompt_tokens * 0.6) if n_assistant > 0 else 0
        self._json(
            200,
            {
                "id": f"chatcmpl-mock-{n_assistant}",
                "object": "chat.completion",
                "created": 0,
                "model": req.get("model", "mock-director"),
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )


def start_server(workspace: str, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the scripted server in a daemon thread. Returns (server, base_url)."""
    _Handler.workspace = workspace
    httpd = ThreadingHTTPServer((host, port), _Handler)
    actual_port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://{host}:{actual_port}/v1"


if __name__ == "__main__":
    import os

    ws = os.getenv("MEDIA_WORKSPACE", "/tmp/creation")
    srv, base = start_server(ws, port=int(os.getenv("MOCK_LLM_PORT", "8123")))
    print(f"mock LLM (scripted storyboard) serving at {base}  workspace={ws}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
