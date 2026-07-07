# ruff: noqa: E402, E501
"""Demo: a multimodal *creation* agent that turns a brief into a short film.

Runs the full ``AgentInteraction`` loop on the ``local_native`` runtime
(pexpect bash on the host -- no container, no GPU). The agent reads the
``video-storyboard`` skill, generates reference assets + per-shot clips with
the media tools, concatenates a final film, and reports the token cost.

Two ways to drive the LLM:

* **Scripted mock (default)** -- no endpoint needed. A tiny OpenAI-compatible
  server (``mock_llm_server.py``) plays a fixed storyboard so the whole loop
  runs end-to-end offline. Great for CI / a laptop with no GPU.
* **Real endpoint** -- set ``BASE_URL`` (+ ``MODEL_NAME`` / ``API_KEY``) to any
  OpenAI-compatible tool-calling server, e.g. vLLM:

      vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes

Backend for the media tools:

* ``MEDIA_BACKEND=mock`` (default) -- offline Pillow/ffmpeg placeholders.
* ``MEDIA_BACKEND=volc`` -- real Volcengine Ark API; also set ``ARK_API_KEY``
  (and optionally ``VOLC_IMAGE_MODEL`` / ``VOLC_VIDEO_MODEL``).

Run (offline, mock everything):

    python examples/creation_agent/demo.py

Run against a real model + real generation:

    BASE_URL=http://localhost:8000/v1 MODEL_NAME=<m> MEDIA_BACKEND=volc ARK_API_KEY=... \
      python examples/creation_agent/demo.py
"""

import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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

# --- workspace + config ----------------------------------------------------
workspace = Path(os.getenv("CREATION_WORKSPACE", str(Path.home() / ".uni-agent" / "app" / "creation" / "workspace")))
workspace.mkdir(parents=True, exist_ok=True)
usage_log = workspace / "usage.jsonl"
usage_log.unlink(missing_ok=True)  # fresh cost ledger per run

media_backend = os.getenv("MEDIA_BACKEND", "mock").lower()
skills_dir = Path(os.getenv("CREATION_SKILLS_DIR", str(REPO_ROOT / "skills")))

user_request = os.getenv(
    "CREATION_USER_REQUEST",
    "Make a short cinematic clip: a lone astronaut exploring a red alien desert at dusk. "
    "Keep the astronaut consistent across shots.",
)

# --- LLM endpoint: real if BASE_URL set, else start the scripted mock -------
base_url = os.getenv("BASE_URL")
using_mock_llm = base_url is None
mock_server = None
if using_mock_llm:
    from examples.creation_agent.mock_llm_server import start_server

    os.environ["MEDIA_WORKSPACE"] = str(workspace)
    mock_server, base_url = start_server(str(workspace))
    model_name = "mock-director"
    api_key = "EMPTY"
else:
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder")
    api_key = os.getenv("API_KEY", "EMPTY")

print("=" * 80)
print("Multimodal creation agent")
print("=" * 80)
print(f"Run ID:          {run_id}")
print(f"Workspace:       {workspace}")
print(f"Media backend:   {media_backend}")
print(f"LLM endpoint:    {base_url}  ({'scripted mock' if using_mock_llm else 'real'})")
print(f"Model name:      {model_name}")
print(f"Skills dir:      {skills_dir} (exists={skills_dir.is_dir()})")
print(f"User request:    {user_request}")

# --- deployment (local_native: pexpect bash on the host) -------------------
deployment_config = {"type": "local_native", "startup_timeout": 60.0}
env_variables = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    # so the media CLIs (copied onto PATH) can import uni_agent + write the ledger
    "PYTHONPATH": str(REPO_ROOT),
    "MEDIA_BACKEND": media_backend,
    "MEDIA_USAGE_LOG": str(usage_log),
}
for k in ("ARK_API_KEY", "VOLC_API_KEY", "VOLC_IMAGE_MODEL", "VOLC_VIDEO_MODEL"):
    if os.getenv(k):
        env_variables[k] = os.environ[k]

env_config = AgentEnvConfig(
    deployment=deployment_config,
    env_variables=env_variables,
    post_setup_cmd=f"mkdir -p {workspace} && cd {workspace}",
    tool_install_dir=Path("~/.uni-agent/bin").expanduser(),
)
env = AgentEnv(run_id=run_id, env_config=env_config)

# --- tools -----------------------------------------------------------------
tool_names = [
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
tools_manager = ToolsManager(ToolsManagerConfig(tools=[ToolConfig(name=n) for n in tool_names]))

# --- skills ----------------------------------------------------------------
skills_manager = SkillsManager.from_config(SkillsManagerConfig(skills_dir=skills_dir))
print(f"[skills] discovered {len(skills_manager.skills)}: {[s.name for s in skills_manager.skills]}")

# --- model -----------------------------------------------------------------
model = OpenAICompatibleChatModel(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    sampling_params={"temperature": 0.7, "top_p": 0.95},
)
model.set_tools_schemas(tools_manager.tools_schemas)

SYSTEM_PROMPT = f"""You are a multimodal media-creation agent. You fulfil the user's creative request using your generation tools. There is NO fixed pipeline: read the request, decide which tools to use and in what order, and adapt the plan to what the user actually asked for (a single image, a one-shot clip, a multi-shot film, an edit of a provided asset, etc.).

# Workspace
Do all work under: {workspace}
Write every asset there and use the exact paths the tools report back.

# Tools and skills
Your generation tools are text2image, image2image, text2video, image2video, ref2video, and concat_video; you also have media_usage (report cost), video_task (query/cancel an async job), and execute_bash / str_replace_editor / finish. Each generation tool has a matching *skill* (listed under <available_skills>) documenting when and how to use it. Before you use a tool you are unsure about, read its SKILL.md first (e.g. `cat <location>`) so you use the right parameters. Pick tools by the task, not by a script.

# Cost
Every generation spends tokens (see each tool's `usage`). Cost is an evaluation metric -- minimize it: use the smallest size/resolution and shortest duration that meet the request, don't regenerate assets you already have, and don't request extras the user didn't ask for. Call media_usage before finishing and report the total.

# Discipline
- Every assistant response MUST contain EXACTLY ONE tool call.
- End by calling `finish` with a short summary (what you produced, where it is, and the total cost).
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_request},
]

interaction = AgentInteraction(
    run_id=run_id,
    env=env,
    model=model,
    tools_manager=tools_manager,
    messages=messages,
    skills_manager=skills_manager,
    action_timeout=180,
    max_turns=30,
    chat_mode=True,
)

# --- run -------------------------------------------------------------------
print("\n[1/4] Starting environment...")
env.start()

print("[2/4] Installing tools + skills...")
env.install_tools(tools_manager.tools)
env.install_skills(skills_manager)
interaction.inject_skills_manifest()

print("[3/4] Running interaction loop...\n")
result = interaction.run()
trajectory = result["trajectory"]

print("\n[4/4] Result")
last = trajectory[-1] if trajectory else None
print(f"  steps:       {len(trajectory)}")
print(f"  exit_reason: {last.exit_reason if last else '(none)'}")
if last and last.tool_results:
    print(
        "  final tool observation:\n"
        + "\n".join("    " + ln for ln in (last.tool_results[-1].observation or "").splitlines()[:12])
    )

# --- cost summary ----------------------------------------------------------
try:
    from uni_agent.tools.media_gen import mediakit

    totals = mediakit.summarize_usage(usage_log)
    print("\n[cost] usage ledger totals:")
    print(
        f"  calls={totals['calls']}  images={totals['images_generated']}  video_seconds={totals['video_seconds']}  total_tokens={totals['total_tokens']}"
    )
    print(f"  by_tool={totals['by_tool']}")
except Exception as e:  # noqa: BLE001
    print(f"[cost] could not summarize usage: {e}")

final_film = workspace / "final.mp4"
print(
    f"\nFinal film: {final_film} (exists={final_film.is_file()}, bytes={final_film.stat().st_size if final_film.is_file() else 0})"
)

env.close()
if mock_server is not None:
    mock_server.shutdown()
print("[done]")
