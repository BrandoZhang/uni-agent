# `media_creation` — a cost-aware multimodal creation task

A [`Task`](../../uni_agent/tasks/media_creation/) that turns a creative brief into
media (image / video / audio) with the standalone
**[`media-ai`](https://github.com/BrandoZhang/media-ai)** CLI, and scores the result
with a **cost-aware reward**. It plugs into the same task/agent/sandbox machinery as
`swe_bench`, and runs **fully offline with no GPU** via a scripted mock LLM.

## How it fits the architecture

| Layer | What this task uses |
|---|---|
| **Task** | `uni_agent/tasks/media_creation/` — uploads media-ai's skills into the sandbox, injects a manifest into the prompt, runs the agent, scores the reward. |
| **Agent** | the stock **`react`** agent — no bespoke media agent needed. |
| **Tool** | the stock **`shell`** tool (`stateful_shell`) — `media-ai` is a self-driving CLI, so no per-command tool is needed. |
| **Sandbox** | any provider (`local` for the demo; `docker`/`modal` for isolated rollouts). |
| **Skills** | media-ai's own packaged [Agent Skills](https://github.com/BrandoZhang/media-ai/tree/main/skills), located via `media_ai.agent_skills_dir()`, uploaded into the sandbox, and surfaced as a **progressive-disclosure** `<available_skills>` manifest. The agent `cat`s a `SKILL.md` before running the matching command. |
| **Reward** | `reward.py` — `quality_proxy − cost_weight × total_tokens`, clamped to [-1, 1]. Cost is summed from the media-ai `usage` blocks in the episode transcript (never a shared ledger, so it's concurrency-safe); the final film is the last `concat` output, verified to exist in the sandbox. |

The old GUI / per-run workspace-GC from the previous architecture are gone: the new
per-sandbox model isolates each rollout for free, and cost now lives in the reward
(derived from the transcript) rather than a bespoke metering hook.

## Run it

### 1. Offline, end-to-end (no model, no GPU)

Drives the **real** ReAct loop + shell tool + local sandbox + `media-ai`, with a
tiny scripted mock LLM standing in for the policy:

```bash
pip install -e ../media-ai        # the media-ai CLI on PATH (Pillow + ffmpeg pulled in)
pip install aiohttp pyyaml        # uni_agent runtime deps for this path
python examples/media_creation/demo.py
```

Prints the reward and cost, and writes `final.mp4` under `/tmp/media_creation/<run>/`.

### 2. No-LLM toolchain smoke test

Exercises the `media-ai` CLI pipeline directly (no agent, no model):

```bash
python examples/media_creation/smoke_test.py
```

### 3. Real model + real generation

Point at any OpenAI-compatible **tool-calling** endpoint and select a real provider
(`MEDIA_PROVIDER=volc|openai|gemini|elevenlabs` + its `*_API_KEY` in the shell tool's
`env_vars`). Per-sample briefs ride in on the dataset row's `prompt`/`metadata`:

```bash
python examples/inference/parallel_infer_api.py \
    --data-path <briefs.parquet> \
    --task-config examples/media_creation/task_config.yaml \
    --base-url <endpoint> --model <model> --api-key <key> --concurrency 8
```

## Files

```
examples/media_creation/
├── README.md            ← this file
├── demo.py              ← offline end-to-end run (real task/agent, mock LLM)
├── mock_llm_server.py   ← scripted OpenAI-compatible server (no GPU)
├── smoke_test.py        ← no-LLM media-ai CLI pipeline test
└── task_config.yaml     ← config for the parallel_infer_api / RL runner

uni_agent/tasks/media_creation/
├── task.py    ← MediaCreationTask + config (prompt, skills, run, score)
├── reward.py  ← cost-aware reward (cost from the transcript, film checked in the sandbox)
└── skills.py  ← discover media-ai's skills, upload them, build the manifest
```

## Requirements

- **media-ai on PATH** in the sandbox (and importable on the host, for skill discovery).
  Not yet on PyPI; for sandbox dev install it editable: `pip install -e ../media-ai`.
- The offline `mock` provider needs no credentials; real providers read their key from
  the shell tool's `env_vars`.
