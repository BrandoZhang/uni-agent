# Image Agent — a minimal multi-modality media-generation agent

A minimal demo of a **multi-modality media-generation agent** on Uni-Agent: a VLM
orchestrator that writes an image prompt, calls a **frozen** image generator, **sees the
result**, refines, and iterates. It demonstrates **inference** and **chaining** end-to-end,
and ships a GRPO training recipe for the orchestration policy — with an honest, code-referenced
path to *true* pixels-in-the-loss VLM training.

The image generator is [`media-ai`](../../../media-ai); its default `mock` provider is fully
offline, so the demo runs with **no API keys, no network, no GPU**.

> **Read this first — what is (and isn't) trained.**
> The runnable recipe here (`train.sh`, classic `UniAgentLoop`) trains the **orchestrator's
> text/tool-use policy** of a VLM checkpoint: *which* tool to call, *what* prompt to write,
> *when* to stop, scored by a reward. It does **not** feed the generated pixels into the loss —
> `UniAgentLoop.convert_to_agent_output` emits `multi_modal_data={}`, and the training rollout
> uses a text tokenizer (not an image processor), so the **vision encoder gets no gradient**.
> In other words, on this path it is effectively **LLM-style RL over a VLM checkpoint**, not
> learning-from-images. True VLM-from-pixels RL is the **gateway/framework path** (§5) — it is
> real in this repo but needs the full verl rollout stack (vLLM/CUDA), so it is documented and
> CI-scoped here, not run by the offline demo.

## What it adopts from the verl recipes

After `git submodule update --init --recursive` the recipes live under `verl/recipe/`:

| Concern | Adopted from | How it shows up here |
| --- | --- | --- |
| Multi-turn agent + **image fed back as an observation** | `recipe/deepeyes` (`ToolResponse(image=...)`, `tool_agent_loop`) | `AgentInteraction(multimodal_observations=True)` attaches each generated image as an OpenAI `image_url` block so a served VLM sees it next turn |
| **Composite reward** (`0.8·acc + 0.2·format + 1.2·tool`) | `recipe/deepeyes/deepeyes.py::compute_score` | `uni_agent/reward/image_reward.py` (`artifact`/`format`/`tool`/`quality`) |
| **Reward model on generated media** | `recipe/dance_grpo` (HPS reward) + DeepEyes VLM-judge | `uni_agent/reward/quality_models.py` (`vlm_judge` / `image_reward` / `hpsv3`) |
| Prompt-list dataset | `recipe/dance_grpo/data/prompt.txt` | `examples/data_preprocess/image_agent.py` |
| GRPO ("better or worse than siblings") | deepeyes / dance_grpo | `algorithm.adv_estimator=grpo` in `train.sh` |

The tool is the media-generation analog of DeepEyes' `image_zoom_in_tool`: a frozen visual
tool the VLM calls, whose image is fed back — with "crop" replaced by "generate".

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
Composite reward: 2.000  components={'artifact': 1.0, 'format': 0.0, 'tool': 1.0, 'quality': 0.0}
```

### With a real VLM

Serve any OpenAI-compatible vision model, then point the demo at it — now the model actually
reasons over the image it is shown and decides how to refine:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
BASE_URL=http://localhost:8000/v1 MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
    python examples/image_agent/demo.py
```

Use a real generator (not the offline mock) with `MEDIA_PROVIDER=volc|openai|gemini` (plus that
provider's credentials) — no code change.

## 2. The multimodal linchpin (inference)

`AgentInteraction(multimodal_observations=True)` (default **off**, so every existing agent/test
is unchanged) detects the `image_path` in a tool's JSON observation, reads the PNG, and appends
it to the model-visible messages as a base64 `image_url` block. On the inference path
(`OpenAICompatibleChatModel`, which re-sends the messages each turn) the VLM literally sees the
image it just generated — DeepEyes' "thinking with images" on Uni-Agent's own stack.

## 3. Reward

`uni_agent/reward/image_reward.py` (`image_agent`), a DeepEyes-style composite:

- **artifact** (0.8): the PNG exists, is non-empty, and its dimensions match the requested size;
- **format** (0.2, penalty): the rollout stayed well-formed and reached `finish`;
- **tool** (1.2): `generate_image` was used *and* produced a valid artifact;
- **quality** (0.5): perceptual/alignment quality of the final image —
  - **offline default:** a deterministic keyword-overlap proxy vs. `ground_truth.keywords`
    (enough to make the demo/tests reproducible; **not** a real quality signal);
  - **real reward model:** set `reward.quality_model` to one of `uni_agent/reward/quality_models.py`:
    - `vlm_judge` — a served VLM rates the image 1–10 (portable, no extra weights; the DeepEyes
      judge pattern). Needs a served VLM endpoint (`JUDGE_BASE_URL`/`JUDGE_MODEL_NAME`).
    - `image_reward` — ImageReward preference model (`pip install image-reward`).
    - `hpsv3` — HPSv3 human-preference score (`pip install hpsv3` + weights).

> **HPSv3 / any quality model is meaningless on the `mock` provider** — every mock image is a
> placeholder card, so a perceptual model scores noise. The quality slot is therefore **off by
> default**; enable it only with a real `MEDIA_PROVIDER`. Enable via agent_config:
> ```yaml
> reward:
>   name: image_agent
>   quality_model: { name: vlm_judge, base_url: http://localhost:8000/v1, model: Qwen/Qwen2.5-VL-7B-Instruct }
> ```

Unit tests (offline): `pytest tests/reward/test_image_reward.py tests/reward/test_quality_models.py`
(covers the composite, the real-scorer path via a stub, graceful fallback, and VLM-judge parsing).

## 4. Training the orchestration policy (runnable recipe)

```bash
# a) build the dataset (intent prompts -> parquet; agent_name="image_agent")
python examples/data_preprocess/image_agent.py --local_save_dir ~/uni-agent/data/image_agent

# b) launch GRPO (needs a GPU node + a VLM; edit paths in the script)
bash examples/image_agent/train.sh
```

`agent_config.yaml` wires `env` (host) + `tools: [generate_image, finish]` + `reward: image_agent`;
its `name` must equal the dataset's `agent_name`. `train.sh` runs verl's
`fully_async_policy.fully_async_main` with `adv_estimator=grpo`, `rollout.mode=async`,
`multi_turn.enable=True`, `agent.agent_loop_config_path=agent_config.yaml`. **This trains the
orchestrator's text/tool policy (see the box at the top), not the vision path.**

## 5. True VLM-from-pixels RL — the gateway/framework path (documented, CI-scoped)

To put the generated pixels *into the training loss* (vision encoder gets gradients), use
Uni-Agent's gateway/framework stack instead of `UniAgentLoop`. It already carries images into
the training tensors — this is not hypothetical, it is exercised by
`tests/uni_agent/framework/test_generate_sequences_on_cpu.py` and
`tests/uni_agent/framework/test_multi_modal_postprocess_on_cpu.py`. **It requires the full verl
rollout stack (vLLM/CUDA), which the offline demo env cannot install, so the steps below are
documented and validated in CI, not run here.**

The contract (verified by reading the code):

1. **Runner** — implement `AgentRunner`
   (`uni_agent/framework/framework.py`): `async def __call__(*, session, raw_prompt, sample_index, **kw)`.
   Loop: build an OpenAI chat payload, `await session.run_generation(payload, backend)`, execute
   `generate_image`, then append the produced image to the next message as an
   `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` content block.
2. **Images → tensors (automatic)** — the gateway codec
   (`uni_agent/gateway/session/codec.py::extract_multi_modal_data`) scans messages for
   `image_url`/`video_url` parts and carries the images into (a) backend generation (the VLM sees
   them) and (b) `Trajectory.multi_modal_data`;
   `uni_agent/framework/multi_modal_postprocess.py` (`compute_multi_modal_inputs` +
   `compute_position_ids`) runs the HF processor + `get_rope_index` to produce
   `pixel_values`/`image_grid_thw`/mrope position ids — so **image tokens enter the loss**.
3. **Reward** — reuse `uni_agent/reward/image_reward.py` (with a real `quality_model`) or attach
   per-session reward via `session.set_reward_info(...)`.
4. **Trainer wiring** —
   `actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter`
   and register the runner under `actor_rollout_ref.rollout.custom.agent_framework.agent_runners.image_agent`
   with a VLM `actor_rollout_ref.model.path` (so a processor is loaded).
5. **CPU validation** — mirror `tests/uni_agent/framework/test_generate_sequences_on_cpu.py`
   (`FakeProcessor`, mock backend from `tests/uni_agent/support.py`) to assert the produced
   `Trajectory.multi_modal_data` + `multi_modal_inputs` tensors without a GPU.

## Files

| Path | Purpose |
| --- | --- |
| `uni_agent/tools/generate_image/` | the frozen image tool (schema + media-ai wrapper script) |
| `uni_agent/reward/image_reward.py` | the composite `image_agent` reward |
| `uni_agent/reward/quality_models.py` | pluggable real quality reward (vlm_judge / image_reward / hpsv3) |
| `uni_agent/interaction/interaction.py` | opt-in `multimodal_observations` image feedback |
| `examples/image_agent/demo.py` | inference + chaining demo (offline stub or served VLM) |
| `examples/image_agent/agent_config.yaml` | UniAgentLoop wiring (tools + reward + env) |
| `examples/image_agent/runtime_env.yaml` | Ray runtime env |
| `examples/image_agent/train.sh` | fully-async GRPO launcher (orchestration policy) |
| `examples/data_preprocess/image_agent.py` | intent prompts → parquet |
