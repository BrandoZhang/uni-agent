# ruff: noqa: E501
"""Minimal multi-modality image-generation agent -- inference + chaining demo.

Run the agent (a VLM) that, turn by turn:
  1. writes an image prompt and calls the ``generate_image`` tool (media-ai, offline
     ``mock`` provider by default),
  2. SEES the generated image fed back as a multimodal observation
     (``AgentInteraction(multimodal_observations=True)`` -- the DeepEyes image-feedback
     mechanism realized on Uni-Agent's own stack),
  3. refines the prompt and regenerates (chaining), then calls ``finish``.

Two ways to run:

  # (a) Fully offline smoke test -- zero external services, a scripted stub stands in
  #     for the VLM just to exercise the whole loop (tool-calling, image feedback,
  #     chaining, reward). No network, no GPU, no API keys.
  OFFLINE=1 python examples/image_agent/demo.py

  # (b) Real VLM -- point at any OpenAI-compatible endpoint serving a vision model
  #     (e.g. Qwen2.5-VL via vLLM). The model actually reasons over the image it sees.
  BASE_URL=http://localhost:8000/v1 MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
      python examples/image_agent/demo.py
"""

import json
import os
import uuid
from pathlib import Path

from uni_agent.interaction import (
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    OpenAICompatibleChatModel,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.reward.image_reward import score_trajectory
from uni_agent.tools import ToolConfig

OFFLINE = os.getenv("OFFLINE", "0").lower() in ("1", "true", "yes")

run_id = str(uuid.uuid4())
out_dir = Path(os.getenv("IMAGE_AGENT_OUTPUT_DIR", f"/tmp/image_agent/{run_id}"))
out_dir.mkdir(parents=True, exist_ok=True)

user_request = (
    "Create a poster-style image of a vintage red bicycle leaning against a brick wall "
    "at sunset. Generate a first draft, look at it, refine the prompt once to improve the "
    "composition and lighting, then finish."
)

SYSTEM_PROMPT = (
    "You are an image-generation agent. Every assistant turn MUST contain EXACTLY ONE tool call. "
    "Use `generate_image` to render a prompt; you will then see the image and can refine the prompt "
    "and call `generate_image` again, or call `finish` when satisfied. Do not reply with plain text "
    "without a tool call."
)


def _tool_call(name: str, args: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class StubVLM:
    """Scripted stand-in for a served VLM so the demo runs with no external services.

    It does not actually reason about the pixels; it drives a representative
    generate -> (see image) -> refine -> generate -> finish trajectory to exercise the
    full loop offline. Swap in ``OpenAICompatibleChatModel`` for real VLM reasoning.
    """

    def __init__(self) -> None:
        self._turn = 0
        self.tools_schemas: list[dict] | None = None

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages):
        return {"metrics": {}}

    async def append_messages_to_rollout_cache(self, new_messages, rollout_cache):
        return rollout_cache

    async def query(self, messages, rollout_cache, **kwargs):
        self._turn += 1
        if self._turn == 1:
            content = "<think>Drafting the poster with the key elements.</think>"
            calls = [
                _tool_call(
                    "generate_image",
                    {
                        "prompt": "A vintage red bicycle leaning against a weathered brick wall at sunset, warm golden light, poster style",
                        "size": "512x512",
                    },
                )
            ]
        elif self._turn == 2:
            content = "<think>Now I can see the draft; I'll enrich the lighting and composition and regenerate.</think>"
            calls = [
                _tool_call(
                    "generate_image",
                    {
                        "prompt": "A vintage red bicycle against a brick wall, dramatic sunset, long shadows, cinematic composition, rich warm tones, high detail poster",
                        "size": "512x512",
                    },
                )
            ]
        else:
            content = "<think>The refined poster looks good.</think>"
            calls = [_tool_call("finish", {"answer": "Delivered a sunset red-bicycle poster after one refinement."})]
        return content, calls, rollout_cache, {"prompt_tokens": 0, "completion_tokens": len(content)}


print("=" * 80)
print("Uni-Agent multi-modality image-generation agent demo")
print("=" * 80)
print(f"Run ID: {run_id}")
print(f"Mode: {'OFFLINE (scripted stub VLM, mock image provider)' if OFFLINE else 'served VLM'}")
print(f"Image output dir: {out_dir}")

# Env: run tools directly on the host (no container / GPU). The image provider is
# media-ai's offline `mock` unless $MEDIA_PROVIDER says otherwise.
env = AgentEnv(
    run_id=run_id,
    env_config=AgentEnvConfig(
        deployment={"type": "host"},
        env_variables={
            "IMAGE_AGENT_OUTPUT_DIR": str(out_dir),
            "MEDIA_PROVIDER": os.getenv("MEDIA_PROVIDER", "mock"),
            "MEDIA_USAGE_LOG": str(out_dir / "media_usage.jsonl"),
        },
    ),
)

tools_manager = ToolsManager(
    ToolsManagerConfig(
        tools=[ToolConfig(name="generate_image"), ToolConfig(name="finish")],
        parser="hermes",
    )
)

if OFFLINE:
    model = StubVLM()
else:
    model = OpenAICompatibleChatModel(
        base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("API_KEY", "EMPTY"),
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"),
        sampling_params={"temperature": 0.7, "max_tokens": 2048},
    )
model.set_tools_schemas(tools_manager.tools_schemas)

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
    action_timeout=120,
    max_turns=8,
    multimodal_observations=True,  # <-- the VLM sees each generated image next turn
)


async def main() -> None:
    # Run the whole session in ONE event loop: the host deployment holds a persistent
    # bash subprocess bound to this loop, so start/install/run/close must share it
    # (auto_await returns the coroutine to await when the caller is async).
    print("\n[1/4] Starting environment...")
    await env.start()

    print("[2/4] Installing tools...")
    await env.install_tools(tools_manager.tools)
    print((await env.communicate("which generate_image && which finish")).strip())

    print("\n[3/4] Running interaction...")
    try:
        result = await interaction.run()
    finally:
        await env.close()

    print("\n[4/4] Trajectory summary:")
    generated_images: list[str] = []
    for step in result["trajectory"]:
        tools = ", ".join(f"{tr.name}[{tr.status}]" for tr in step.tool_results) or "(no tool)"
        print(f"  step {step.step_idx}: exit={step.exit_reason} tools={tools}")
        for tr in step.tool_results:
            if tr.name == "generate_image" and tr.status == "ok":
                try:
                    info = json.loads(tr.observation[tr.observation.find("{") : tr.observation.rfind("}") + 1])
                    if info.get("image_path"):
                        generated_images.append(info["image_path"])
                except (json.JSONDecodeError, ValueError):
                    pass

    # Count the multimodal observations that were fed back to the VLM (image_url blocks).
    shown_images = sum(
        1
        for m in interaction.messages
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )

    score, reward_info = score_trajectory(result["trajectory"])
    print(f"\nGenerated images ({len(generated_images)}):")
    for p in generated_images:
        print(f"  - {p}")
    print(f"Images shown back to the VLM (multimodal feedback): {shown_images}")
    print(f"\nComposite reward: {score:.3f}  components={reward_info['components']}")
    print("\nDone.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
