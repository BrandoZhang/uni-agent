# `creation_agent` — a multimodal video-creation agent (tiny demo)

A director agent that turns a creative brief into a short film: it plans a
storyboard, locks reference assets for cross-shot consistency, generates
each shot, concatenates a final film, and reports the **token cost**. Built
on the same `AgentInteraction` loop as the other uni-agent examples.

It runs **fully offline with no GPU** by default (mock generation backend +
a scripted mock LLM), and switches to real models by flipping two env vars.

---

## What this demonstrates (the four requirements)

1. **Agent Skill standard, framework-decoupled.** uni-agent already supports
   the Agent Skills standard natively (`uni_agent/skills/`): a `SKILL.md`
   with YAML frontmatter + progressive disclosure (only a manifest goes in
   the system prompt; the body is `cat`-ed on demand). We did **not** need
   DeepAgents or a Claude Code CLI. There is **one skill per generation
   tool** ([`skills/text2image`](../../skills/text2image/),
   [`image2image`](../../skills/image2image/),
   [`text2video`](../../skills/text2video/),
   [`image2video`](../../skills/image2video/),
   [`ref2video`](../../skills/ref2video/),
   [`concat_video`](../../skills/concat_video/)) documenting *that tool's*
   usage. The agent reads the relevant skill on demand and composes tools
   adaptively per the user's request — there is no hard-coded pipeline.
2. **Full Volcengine image + video capability set**, wrapped in CLIs that
   own the HTTP calls — including *multimodal-reference* video. See the tool
   table below.
3. **Local filesystem only** — the `local_native` deployment runs bash on the
   host (no sandbox needed for a tiny demo).
4. **Cost tracking as an evaluation metric** — every generation writes a
   usage ledger; a cost-aware reward spec turns it into a score to minimize.

---

## Tool suite → Volcengine (Ark) capabilities

Each tool is a command from the standalone **[`media-ai`](../../packages/media-ai/)**
package (`pip install -e packages/media-ai`) — a self-contained CLI toolkit
with **no uni-agent dependency**, so the same tools drop into any agent
framework's sandbox. uni-agent registers them as *system tools* (thin
schema wrappers in [`uni_agent/tools/media_gen/`](../../uni_agent/tools/media_gen/));
the implementation + HTTP calls live in the package's `mediakit.py`. The
default backend is an offline **mock** (Pillow images + ffmpeg clips);
`--backend volc` calls the real Ark API (Bearer API-Key auth).

| Tool | Capability | Ark endpoint |
|---|---|---|
| `text2image`  | text → image, **group images** (`max_images>1`) | `POST /images/generations` |
| `image2image` | 1..N reference images → image(s) | `POST /images/generations` |
| `text2video`  | text → clip | `POST /contents/generations/tasks` |
| `image2video` | first-frame (+ optional **last-frame**) → clip; `return_last_frame` to chain | `POST /contents/generations/tasks` |
| `ref2video`   | **multimodal reference**: images(0-9)+videos(0-3)+audio(0-3)+text → clip (Seedance 2.0) | `POST /contents/generations/tasks` |
| `concat_video`| join per-shot clips → final film | local ffmpeg |
| `video_task`  | query / cancel an async video task (cost control) | `GET`/`DELETE /contents/generations/tasks/{id}` |
| `media_usage` | report accumulated token cost from the ledger | local ledger |

Video generation is an async task; the CLIs **submit + poll to completion**
so the agent sees one synchronous call. Use `video_task --op cancel` to kill
a queued task (a cost lever).

---

## Run it

### 1. No-LLM smoke test (verify the toolchain now)

Needs only the media-ai package (`pip install -e packages/media-ai`, which
pulls in Pillow + ffmpeg). No model, no credentials, no uni-agent loop:

```bash
python examples/creation_agent/smoke_test.py
```

Runs text2image → image2image → image2video → text2video → ref2video →
concat_video → media_usage and asserts every artifact + the cost ledger.

### 2. Full agent loop, offline (scripted mock LLM)

Drives the **real** `AgentInteraction` loop end-to-end with a tiny scripted
OpenAI-compatible server (see the vLLM note below), the `local_native`
runtime, the real tools, and the mock media backend:

```bash
pip install swe-rex openai loguru pydantic pydantic_settings orjson regex
pip install -e packages/media-ai          # the media tools (Pillow + ffmpeg pulled in)
python examples/creation_agent/demo.py
```

Produces `~/.uni-agent/app/creation/workspace/final.mp4` and prints the cost
ledger totals.

### 3. Real model + real generation

Point at any OpenAI-compatible **tool-calling** endpoint and turn on the
Volcengine backend:

```bash
BASE_URL=http://localhost:8000/v1 MODEL_NAME=<model> API_KEY=... \
MEDIA_BACKEND=volc ARK_API_KEY=<ark-key> \
  python examples/creation_agent/demo.py
```

For the Volc backend the Model ID is chosen per call with `--model`
(optional), else `$ARK_IMAGE_MODEL` / `$ARK_VIDEO_MODEL`, else a built-in
default. Models must be enabled for your account (full list:
<https://www.volcengine.com/docs/82379/1330310>):

```bash
export ARK_API_KEY=...                              # long-lived Ark API key
export ARK_IMAGE_MODEL=doubao-seedream-4-5-251128   # optional global default
export ARK_VIDEO_MODEL=doubao-seedance-2-0-260128   # optional global default
```

---

## On running a real vLLM here (follow-up #2)

This sandbox has **no GPU** (4 CPU / 15 GB RAM), so a real vLLM tool-calling
model is impractical. Instead, `demo.py` ships
[`mock_llm_server.py`](./mock_llm_server.py) — a ~120-line stdlib
`http.server` that speaks the OpenAI `/v1/chat/completions` protocol and
returns a **scripted storyboard** of `tool_calls`. That drives the entire
real loop (tool parsing, bash runtime, media CLIs, usage ledger, `finish`)
with zero GPU, which is exactly what you want for CI / a laptop.

To use a real vLLM instead, just set `BASE_URL` (see run #3). A tool-calling
model served like this works out of the box:

```bash
vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

---

## Cost tracking & the reward (follow-up #3)

Every generation appends a line to the usage ledger (`$MEDIA_USAGE_LOG`,
default `<workspace>/usage.jsonl`) with the tool, backend, model, and token
usage. For the mock backend the tokens are **synthesized with the same
formula the Ark docs document** (image `output_tokens = images * ⌊w·h/256⌋`),
so the cost path is exercised offline.

`media_usage` aggregates the ledger; the reward spec
[`uni_agent/reward/media_creation.py`](../../uni_agent/reward/media_creation.py)
turns it into a score:

```
reward = quality_proxy - cost_weight * total_tokens      (clamped to [-1, 1])
```

so a policy that meets the brief with **fewer / cheaper generations** scores
higher. `quality_proxy` is a placeholder (did we produce a real film) — swap
in a VLM-as-judge / aesthetic / preference model for production. The per-tool
skills also tell the agent to minimize cost (smallest size/resolution,
shortest duration, no redundant regens) and the system prompt asks it to
report the total.

---

## Workspace isolation (concurrent / long-horizon tasks)

Creation is IO to a filesystem + HTTP calls — it does **not** need a sandbox
for safety the way arbitrary code execution does — so the local filesystem is
a fine substrate for a workspace. But on a **shared** filesystem
(`local_native` / `host`, where every rollout sees the same host FS) isolation
is *not* automatic: the agent picks its own output names (`ref.png`,
`final.mp4`, …), so two concurrent tasks that pick the same name would
overwrite each other's artifacts and share the cost ledger. (Container
backends — `modal` / `vefaas` / `local` — isolate the FS per run for free.)

So the workspace is keyed by a **`workspace_key`**: the effective directory is
`<base>/<workspace_key>`.

- **What is one "run"?** A *run* is one full trajectory — a task from the
  initial prompt to its terminal state. However many turns it takes (reflect,
  regenerate shot 3, re-cut the film) and even across a **partial-rollout**
  pause/resume, it stays **one `run_id`, one workspace**: the assets
  accumulate in one place, which is exactly what continuity needs. Two
  *different* tasks (or a re-sample of the same prompt) get different keys and
  never collide.

- **Training / offline eval** — one rollout *is* one task's whole lifecycle,
  so `workspace_key` defaults to the per-rollout `run_id` (a fresh uuid per
  run is exactly right). `UniAgentLoop` derives this automatically when
  `env.env_variables.MEDIA_WORKSPACE_BASE` is set (see `config.yaml`); it sets
  `MEDIA_WORKSPACE` + `MEDIA_USAGE_LOG` under `<base>/<run_id>` and `cd`s the
  shell there. No tool changes, no per-task wiring.

- **Interactive, resumable serving** — a real user returning hours later to
  keep editing the *same* project spans many model invocations (many `run_id`s),
  so a per-invocation uuid would hand them an empty workspace and lose prior
  assets. Pin a stable **`MEDIA_WORKSPACE_KEY`** (a conversation / task id from
  the caller); the same key maps to the same dir, so the artifacts persist and
  the task resumes. `demo.py` mirrors this with `CREATION_SESSION_ID`
  (defaulting to `run_id`).

The rule of thumb: the isolation key is **"one per independent task, stable
for that task's whole lifetime"** — `run_id` satisfies that for a one-shot
rollout; a caller-supplied session id satisfies it for a resumable session.
Cost accounting is already concurrency-safe regardless (the reward derives
tokens from *this* run's trajectory, never the shared ledger).

### Reclaiming the workspace at run end (`MEDIA_WORKSPACE_GC`)

On a shared FS the per-run dirs would pile up under `<base>/` (a container
backend reclaims its FS at `env.close()`, so this only bites `local_native` /
`host`). Set **`MEDIA_WORKSPACE_GC=reclaim`** to have `UniAgentLoop` delete
`<base>/<run_id>` at run end (default off, so demo/eval runs keep their
artifacts to inspect; `config.yaml` turns it on for the 64-way training path).

The one thing to get right is **sandbox vs. local FS**: the artifacts live on
the *container's* FS for container backends and on the *host's* FS for
`local_native` / `host`. So the deletion runs **through the env** (`rm -rf`
inside the sandbox), never a host-side `shutil.rmtree` — that targets whichever
FS is real and can't delete a host path for a container run. It fires *after*
the reward (cost + film already extracted) and *before* `env.close()` (while
the shell is alive), and is best-effort (never fails the run). Scope guards:
it only ever deletes a dir this loop derived, and **never** a session-keyed
(resumable) dir. Session-level and TTL/orphan reaping are deliberately left to
the harness layer (a real run may crash before the finally block ever runs).

> Caveat: `MEDIA_WORKSPACE_GC` also removes a *failed* run's artifacts. Keep it
> off if you need to debug failures. And `keep=final`/exporting the film off a
> container both require an explicit copy-out step (the film dies with the
> container otherwise) — that's future harness work, not part of this reclaim.

## Two entrypoints

- **`demo.py`** — standalone `AgentInteraction` run; endpoint via env vars.
  Best for trying it out. Needs no verl.
- **`config.yaml`** — the same tools + skill + cost-aware reward wired into
  `uni_agent.agent_loop.UniAgentLoop`, i.e. the parallel-inference / RL
  training entrypoint. This is the bridge from "run it" to "train it".

---

## Files

```
examples/creation_agent/
├── README.md            ← this file
├── demo.py              ← standalone agent run (mock LLM by default)
├── mock_llm_server.py   ← scripted OpenAI-compatible server (no GPU)
├── smoke_test.py        ← no-LLM toolchain test
└── config.yaml          ← UniAgentLoop (large-scale / training) config

packages/media-ai/          ← standalone media CLI toolkit (mock + Ark backends)
uni_agent/tools/media_gen/   ← thin uni-agent registrations for those commands
uni_agent/reward/media_creation.py  ← cost-aware reward spec
skills/<tool>/SKILL.md       ← one skill per generation tool (usage guidance)
```
