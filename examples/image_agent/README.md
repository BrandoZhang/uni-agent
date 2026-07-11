# Image Agent — a minimal multi-modality media-generation agent

A minimal demo of a **multi-modality media-generation agent** on Uni-Agent: a VLM
orchestrator that writes an image prompt, calls a **frozen** image generator, **sees the
result**, refines, and iterates. It demonstrates **inference** and **chaining**, and ships
the scaffolding to make it **reproducible for multi-modality VLM RL training**.

The image generator is [`media-ai`](../../../media-ai) (a provider-agnostic media CLI);
its default `mock` provider is fully offline, so the whole demo runs with **no API keys,
no network, no GPU**.

## What it adopts from the verl recipes

Read after `git submodule update --init --recursive` (the recipes live under `verl/recipe/`):

| Concern | Adopted from | How it shows up here |
| --- | --- | --- |
| Multi-turn VLM agent + **image fed back as an observation** | `recipe/deepeyes` (`ToolResponse(image=...)`, `tool_agent_loop`) | `AgentInteraction(multimodal_observations=True)` attaches each generated image as an OpenAI `image_url` block so a served VLM sees it next turn |
| **Composite reward** (`0.8·acc + 0.2·format + 1.2·tool`) | `recipe/deepeyes/deepeyes.py::compute_score` | `uni_agent/reward/image_reward.py` (`artifact`/`format`/`tool`/`align`) |
| **Reward model on generated media** + prompt-list dataset | `recipe/dance_grpo` (HPS reward, `data/prompt.txt`) | the `align` slot (swap in HPSv3/ImageReward/CLIP/VLM-judge) + `examples/data_preprocess/image_agent.py` |
| GRPO for "did this rollout get better or worse" | deepeyes / dance_grpo | `algorithm.adv_estimator=grpo` in `train.sh` (group-relative baseline) |

The tool is the media-generation analog of DeepEyes' `image_zoom_in_tool`: same idea (a
frozen visual tool the VLM calls and whose image is fed back), with "crop" replaced by
"generate".

## Setup

```bash
# from the uni-agent repo root
git submodule update --init --recursive          # brings verl + all recipe/*
pip install --no-deps -e ./verl
pip install -e .                                  # uni_agent importable
pip install -e /path/to/media-ai                  # the frozen image generator
pip install swe-rex loguru pydantic pydantic-settings orjson openai requests regex pillow
```

## 1. Inference + chaining (offline, no GPU)

```bash
OFFLINE=1 python examples/image_agent/demo.py
```

A scripted stub stands in for the VLM and drives a representative trajectory:
`generate_image` → (sees the image) → refine → `generate_image` → `finish`. Expected tail:

```
step 1: exit=completed tools=generate_image[ok]
step 2: exit=completed tools=generate_image[ok]
step 3: exit=finished tools=finish[ok]
Images shown back to the VLM (multimodal feedback): 2
Composite reward: 2.000  components={'artifact': 1.0, 'format': 0.0, 'tool': 1.0, 'align': 0.0}
```

### With a real VLM

Serve any OpenAI-compatible vision model, then point the demo at it — now the model
actually reasons over the image it is shown and decides how to refine:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
BASE_URL=http://localhost:8000/v1 MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
    python examples/image_agent/demo.py
```

Use a real generator (instead of the offline mock) with `MEDIA_PROVIDER=volc|openai|gemini`
(plus that provider's credentials) — no code change.

## 2. The multimodal linchpin

`AgentInteraction(multimodal_observations=True)` (default **off**, so every existing
agent/test is unchanged) detects the `image_path` in a tool's JSON observation, reads the
PNG, and appends it to the model-visible messages as a base64 `image_url` block. On the
inference path (`OpenAICompatibleChatModel`, which re-sends the messages each turn) the VLM
literally sees the image it just generated — this is DeepEyes' "thinking with images",
realized on Uni-Agent's own stack.

## 3. Reward

`uni_agent/reward/image_reward.py` (`image_agent`) is a DeepEyes-style composite, computed
fully offline:

- **artifact** (0.8): the PNG exists, is non-empty, and its dimensions match the requested size;
- **format** (0.2, penalty): the rollout stayed well-formed and reached `finish`;
- **tool** (1.2): `generate_image` was used *and* produced a valid artifact;
- **align** (0.5, optional): keyword overlap vs. `ground_truth.keywords` — **the slot for a
  real reward model** (HPSv3 / ImageReward / CLIPScore / VLM-as-judge, à la DanceGRPO/DeepEyes).

Unit tests: `pytest tests/reward/test_image_reward.py`.

## 4. Training (reproducible scaffolding)

```bash
# a) build the dataset (intent prompts -> parquet; agent_name="image_agent")
python examples/data_preprocess/image_agent.py --local_save_dir ~/uni-agent/data/image_agent

# b) launch GRPO (needs a GPU node + a VLM; edit paths in the script)
bash examples/image_agent/train.sh
```

`agent_config.yaml` wires `env` (host) + `tools: [generate_image, finish]` +
`reward: image_agent`; its `name` must equal the dataset's `agent_name`. `train.sh` runs
verl's `fully_async_policy.fully_async_main` with `adv_estimator=grpo`, `rollout.mode=async`,
`multi_turn.enable=True`, `agent.agent_loop_config_path=agent_config.yaml`. See
`examples/agent_train/single_node_debug.sh` for the full megatron/cluster knobs.

### Scope note — pixels in the training loss

On this **classic `UniAgentLoop` path**, the orchestrator VLM is trained on the **text /
tool-use tokens**; the generated image is fed back at *inference* but not injected into the
training tensors (`convert_to_agent_output` emits `multi_modal_data={}`). This trains the
"editor/orchestrator" — which tool to call, what prompt to write, when to stop — which is
exactly the target of this demo.

**Next step for pixels-in-the-loss VLM RL:** Uni-Agent's gateway/framework path
(`uni_agent/framework/` + `uni_agent/gateway/`) already carries images into the training
tensors — `gateway/session/codec.py::extract_multi_modal_data` pulls `image_url` blocks into
`Trajectory.multi_modal_data`, and `framework/multi_modal_postprocess.py` runs the HF
processor + `get_rope_index`. Wiring an `image_agent` runner there
(`actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter`)
gives true multi-modal training; it is CPU-validatable by mirroring
`tests/uni_agent/framework/test_generate_sequences_on_cpu.py` (`FakeProcessor`).

## Files

| Path | Purpose |
| --- | --- |
| `uni_agent/tools/generate_image/` | the frozen image tool (schema + media-ai wrapper script) |
| `uni_agent/reward/image_reward.py` | the composite `image_agent` reward |
| `uni_agent/interaction/interaction.py` | opt-in `multimodal_observations` image feedback |
| `examples/image_agent/demo.py` | inference + chaining demo (offline stub or served VLM) |
| `examples/image_agent/agent_config.yaml` | UniAgentLoop wiring (tools + reward + env) |
| `examples/image_agent/runtime_env.yaml` | Ray runtime env |
| `examples/image_agent/train.sh` | fully-async GRPO launcher |
| `examples/data_preprocess/image_agent.py` | intent prompts → parquet |
