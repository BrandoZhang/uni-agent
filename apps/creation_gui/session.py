# ruff: noqa: E501
"""Per-session runner: build an ``AgentInteraction`` and drive it live.

This is the **only** module that imports uni-agent. It runs the real loop by
calling the *public* ``AgentInteraction.step()`` in its own small loop (option
A' — uni-agent core is untouched, no per-step hook added to the framework),
invoking ``on_step(step_output, metrics)`` after every step so the server can
translate + stream events. Swap uni-agent for another framework by rewriting
just this file; the GUI and the event/artifact helpers stay framework-neutral.

Offline by default: if no ``BASE_URL`` is given it starts the scripted mock LLM
(``examples/creation_agent/mock_llm_server``) bound to this session's own
workspace, and the media tools default to the offline mock backend.
"""

from __future__ import annotations

import os
import sys
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TOOL_NAMES = [
    "execute_bash",
    "str_replace_editor",
    "text2image",
    "image2image",
    "text2video",
    "image2video",
    "ref2video",
    "concat_video",
    "video_task",
    "media_usage",
    "finish",
]

_SYSTEM_PROMPT = """You are a multimodal media-creation agent. You fulfil the user's creative request using your generation tools. There is NO fixed pipeline: read the request, decide which tools to use and in what order, and adapt to what the user actually asked for.

# Workspace
Do all work under: {workspace}
Write every asset there and use the exact paths the tools report back.

# Tools and skills
Your generation tools are text2image, image2image, text2video, image2video, ref2video, and concat_video; you also have media_usage, video_task, execute_bash, str_replace_editor and finish. Each generation tool has a matching skill (under <available_skills>); read it on demand before using a tool you're unsure about.

# Cost
Every generation spends tokens. Minimize it: smallest size/resolution and shortest duration that meet the request, no redundant regenerations. Call media_usage before finishing and report the total.

# Discipline
- Every assistant response MUST contain EXACTLY ONE tool call (unless you emit several deliberately in one turn).
- End by calling `finish` with a short summary (what you produced, where, and the total cost).
"""


def _start_mock_llm(workspace: str) -> tuple[ThreadingHTTPServer, str]:
    """Start the scripted mock LLM bound to *this* session's workspace.

    Uses a per-session Handler subclass so concurrent sessions don't clobber
    the shared ``_Handler.workspace`` class attribute.
    """
    from examples.creation_agent.mock_llm_server import _Handler

    handler_cls = type("SessionMockHandler", (_Handler,), {"workspace": workspace})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def run_session(
    *,
    brief: str,
    workspace: str | Path,
    backend: str = "mock",
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    skills_dir: str | Path | None = None,
    on_step,
    should_cancel,
) -> None:
    """Build and drive one creation run to completion (blocking; run in a thread).

    ``on_step(step_output, metrics)`` fires after each step; ``should_cancel()``
    is polled between steps to allow a cooperative cancel.
    """
    import asyncio

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    usage_log = workspace / "usage.jsonl"
    usage_log.unlink(missing_ok=True)
    skills_dir = Path(skills_dir) if skills_dir else (REPO_ROOT / "skills")

    mock_httpd: ThreadingHTTPServer | None = None
    if not base_url:
        mock_httpd, base_url = _start_mock_llm(str(workspace))
        model_name, api_key = "mock-director", "EMPTY"
    model_name = model_name or "Qwen/Qwen3-Coder"
    api_key = api_key or "EMPTY"

    # Heavy imports are local so the server module + pure helpers import cleanly
    # (and tests run) without the swe-rex / openai stack present.
    from uni_agent.interaction import (
        AgentEnv,
        AgentEnvConfig,
        AgentInteraction,
        OpenAICompatibleChatModel,
        ToolsManager,
        ToolsManagerConfig,
    )
    from uni_agent.skills import SkillsManager, SkillsManagerConfig
    from uni_agent.tools import ToolConfig

    run_id = str(uuid.uuid4())
    env_variables = {"NO_COLOR": "1", "TERM": "dumb", "MEDIA_BACKEND": backend, "MEDIA_USAGE_LOG": str(usage_log)}
    for key in ("ARK_API_KEY", "ARK_IMAGE_MODEL", "ARK_VIDEO_MODEL"):
        if os.getenv(key):
            env_variables[key] = os.environ[key]

    env = AgentEnv(
        run_id=run_id,
        env_config=AgentEnvConfig(
            deployment={"type": "local_native", "startup_timeout": 60.0},
            env_variables=env_variables,
            post_setup_cmd=f"mkdir -p {workspace} && cd {workspace}",
            tool_install_dir=Path("~/.uni-agent/bin").expanduser(),
        ),
    )
    tools_manager = ToolsManager(ToolsManagerConfig(tools=[ToolConfig(name=n) for n in _TOOL_NAMES]))
    skills_manager = SkillsManager.from_config(SkillsManagerConfig(skills_dir=skills_dir))
    model = OpenAICompatibleChatModel(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        sampling_params={"temperature": 0.7, "top_p": 0.95},
    )
    model.set_tools_schemas(tools_manager.tools_schemas)
    interaction = AgentInteraction(
        run_id=run_id,
        env=env,
        model=model,
        tools_manager=tools_manager,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT.format(workspace=workspace)},
            {"role": "user", "content": brief},
        ],
        skills_manager=skills_manager,
        action_timeout=180,
        max_turns=30,
        chat_mode=False,
    )

    async def _main() -> None:
        try:
            await env.start()
            await env.install_tools(tools_manager.tools)
            await env.install_skills(skills_manager)
            interaction.inject_skills_manifest()

            # A': replicate run()'s tiny loop so we can observe each step,
            # without adding a hook to the framework.
            interaction.rollout_cache = await model.prepare_rollout_cache(interaction.messages)
            interaction.trajectory = []
            done = False
            step_idx = 0
            while not done and not should_cancel():
                step_idx += 1
                step_output = await interaction.step(step_idx)
                interaction.trajectory.append(step_output)
                on_step(step_output, dict(interaction.rollout_cache.get("metrics", {})))
                done = step_output.done
                if step_idx >= interaction.max_turns:
                    break
        finally:
            try:
                await env.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the run
                pass

    try:
        asyncio.run(_main())
    finally:
        if mock_httpd is not None:
            mock_httpd.shutdown()
