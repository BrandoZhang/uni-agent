"""A minimal OpenAI-compatible chat-completions server that *scripts* a
storyboard, so the full ReAct loop can be driven end-to-end without a GPU.

There is no GPU in many sandboxes, and a real tool-calling model on CPU is
impractical for a demo. This server stands in for the policy: on each
``POST /v1/chat/completions`` it counts how many assistant turns have already
happened and returns the *next* scripted move (OpenAI ``tool_calls`` shape).
That exercises the real machinery -- the ReAct loop, the ``shell`` tool, the
sandbox, the ``media-ai`` CLI, and the cost-aware reward.

To run against a REAL model instead, ignore this file and point the task's
``agent.model.base_url`` at any OpenAI-compatible tool-calling endpoint.

The script deliberately keeps 2 short 480p shots to keep cost low.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_DEFAULT_WORKSPACE = "/tmp/media_creation"


def _system_content(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "system":
            return str(m.get("content") or "")
    return ""


def _workspace(messages: list[dict]) -> str:
    """Pull the workspace the task told the agent to use out of the system prompt."""
    hit = re.search(r"Do all work under:\s*(\S+)", _system_content(messages))
    return hit.group(1) if hit else _DEFAULT_WORKSPACE


def _skill_location(messages: list[dict], skill: str) -> str | None:
    """Pull a skill's in-sandbox SKILL.md path out of the injected manifest."""
    hit = re.search(rf"<location>([^<]*/{re.escape(skill)}/SKILL\.md)</location>", _system_content(messages))
    return hit.group(1) if hit else None


def _shell(command: str) -> dict:
    return {"name": "shell", "arguments": {"command": command}}


def build_script(messages: list[dict]) -> list[list[dict] | None]:
    """Ordered assistant moves. A list of shell calls, or ``None`` to finish
    (reply with plain text and no tool call, which ends the ReAct episode)."""
    ws = _workspace(messages).rstrip("/")
    shared = _skill_location(messages, "media-ai-shared")
    video = _skill_location(messages, "media-ai-video")
    read = [_shell(f"cat {shared}")] if shared else [_shell(f"ls -la {ws}")]
    read_video = [_shell(f"cat {video}")] if video else [_shell("media-ai --help")]
    return [
        read,  # 0: read the shared contract skill first
        read_video,  # 1: read the video skill
        [  # 2: mint a reference frame for cross-shot consistency
            _shell(
                f"media-ai image generate "
                f"--prompt 'A lone silver-suited astronaut on a red alien dune, cinematic teal-and-orange grade' "
                f"--output {ws}/ref_hero.png --seed 7"
            )
        ],
        [  # 3: two shots in one turn (the harness runs both)
            _shell(
                f"media-ai video generate --first-frame {ws}/ref_hero.png "
                f"--prompt 'the astronaut slowly turns toward camera, gentle push-in' "
                f"--output {ws}/shot1.mp4 --seconds 3 --resolution 480p"
            ),
            _shell(
                f"media-ai video generate "
                f"--prompt 'wide establishing shot of twin suns setting over the alien desert at dusk' "
                f"--output {ws}/shot2.mp4 --seconds 3 --resolution 480p"
            ),
        ],
        [_shell(f"media-ai concat --input {ws}/shot1.mp4 --input {ws}/shot2.mp4 --output {ws}/final.mp4")],  # 4
        [_shell("media-ai usage")],  # 5
        None,  # 6: finish with a plain-text summary
    ]


_FINISH_TEXT = (
    "Done. Storyboard: (1) a video generated from the hero astronaut's locked reference frame, "
    "(2) a text-to-video establishing shot, concatenated into final.mp4. Cross-shot consistency "
    "came from ref_hero.png used as shot 1's first frame. See the `media-ai usage` output for the cost."
)


class _Handler(BaseHTTPRequestHandler):
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
        script = build_script(messages)
        step = script[n_assistant] if n_assistant < len(script) else None

        if step is None:  # plain-text answer, no tool call -> ReAct finishes
            message = {"role": "assistant", "content": _FINISH_TEXT}
        else:
            tool_calls = [
                {
                    "id": f"call_{n_assistant}_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                }
                for i, c in enumerate(step)
            ]
            message = {
                "role": "assistant",
                "content": f"[director step {n_assistant}] {', '.join(c['name'] for c in step)}",
                "tool_calls": tool_calls,
            }

        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        prompt_tokens = prompt_chars // 4
        self._json(
            200,
            {
                "id": f"chatcmpl-mock-{n_assistant}",
                "object": "chat.completion",
                "model": req.get("model") or "mock-director",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 24,
                    "total_tokens": prompt_tokens + 24,
                },
            },
        )


def start_server(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the scripted server in a daemon thread. Returns (server, base_url)."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}/v1"


if __name__ == "__main__":
    import os

    srv, base = start_server(port=int(os.getenv("MOCK_LLM_PORT", "8123")))
    print(f"mock LLM (scripted storyboard) serving at {base}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
