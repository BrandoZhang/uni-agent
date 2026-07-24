# ruff: noqa: E402
"""Demo: the media-creation task, offline, end-to-end, no GPU.

Runs the REAL machinery -- the ``media_creation`` task, a real ``react`` agent,
the ``shell`` tool, a ``local`` sandbox, and the ``media-ai`` CLI -- driven by a
tiny scripted mock LLM (``mock_llm_server.py``) so the whole loop runs without a
model server. The agent reads media-ai's skills, generates a reference frame +
two shots, concatenates a film, and is scored by the cost-aware reward.

    python examples/media_creation/demo.py

Against a real model instead: set BASE_URL (+ MODEL_NAME / API_KEY) to any
OpenAI-compatible tool-calling endpoint. For real generation, install a media-ai
provider key and set MEDIA_PROVIDER=volc|openai|gemini|elevenlabs in the shell
tool's env_vars below.

Requires: media-ai on PATH (`pip install -e ../media-ai`), aiohttp, pyyaml.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from uni_agent.tasks.registry import get_task

RUN_ID = uuid.uuid4().hex[:8]
WORKSPACE = f"/tmp/media_creation/{RUN_ID}"
BRIEF = os.getenv(
    "CREATION_USER_REQUEST",
    "Make a short cinematic clip: a lone astronaut exploring a red alien desert at dusk. "
    "Keep the astronaut consistent across shots.",
)


async def main() -> int:
    base_url = os.getenv("BASE_URL")
    mock_server = None
    if base_url is None:
        from examples.media_creation.mock_llm_server import start_server

        mock_server, base_url = start_server()
        model_name, api_key = "mock-director", "EMPTY"
    else:
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder")
        api_key = os.getenv("API_KEY", "EMPTY")

    media_provider = os.getenv("MEDIA_PROVIDER", "mock")
    print("=" * 78)
    print("media_creation task demo")
    print("=" * 78)
    print(f"run id:         {RUN_ID}")
    print(f"workspace:      {WORKSPACE}")
    print(f"media provider: {media_provider}")
    print(f"LLM endpoint:   {base_url}  ({'scripted mock' if mock_server else 'real'})")
    print(f"brief:          {BRIEF}\n")

    task_config = {
        "name": "media_creation",
        "sandbox": {"provider": "local"},
        "workspace": WORKSPACE,
        "agent": {
            "name": "react",
            "max_steps": 15,
            # Only the shell tool is needed; media-ai is a self-driving CLI.
            "tools": [
                {
                    "name": "stateful_shell",
                    "command_timeout": 300,
                    "env_vars": {
                        "MEDIA_PROVIDER": media_provider,
                        "MEDIA_USAGE_LOG": f"{WORKSPACE}/usage.jsonl",
                    },
                }
            ],
            "model": {
                "base_url": base_url,
                "model_name": model_name,
                "api_key": api_key,
                "max_total_tokens": 200_000,
            },
        },
        "prompt": [{"role": "user", "content": BRIEF}],
    }

    result = await get_task(task_config).run()

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    info = result.info or {}
    print(f"reward:            {result.reward:.4f}")
    print(f"quality_proxy:     {result.accuracy}")
    print(f"generation tokens: {info.get('total_tokens')}  ({info.get('generation_calls')} calls)")
    print(f"video seconds:     {info.get('video_seconds')}")
    print(f"llm tokens:        {info.get('llm_tokens')}  (exit_reason={info.get('exit_reason')})")
    film = info.get("final_film")
    print(f"final film:        {film}  (exists={info.get('final_film_exists')}, bytes={info.get('final_film_bytes')})")

    if mock_server is not None:
        mock_server.shutdown()
    return 0 if info.get("final_film_exists") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
