# Creation Agent — session handoff & design notes

Carry-over context for continuing the **multimodal creation agent** line of work
(the media generation tools, the cost-aware reward, workspace isolation, and the
web GUI). Read this first in a fresh session, then jump into the relevant files.

Branch for this work: **`claude/laughing-lamport-r3ws5m`**. Latest relevant
commit at time of writing: `be0cf93` (decouple media-ai into its own repo).

---

## 1. What exists now (state snapshot)

A brief → short film agent, built on the same `AgentInteraction` loop as the
other uni-agent examples. It plans a storyboard, generates reference assets +
per-shot clips, concatenates a final film, and reports **token cost** as a
metric. Runs **fully offline, no GPU** by default (scripted mock LLM + mock
media backend); flips to real models via env vars.

Delivered (all on the branch above, all tests green — 118 across the media/
reward/workspace/GUI suites; the media-ai package's own 46 tests live in the
external repo now):

- **`media-ai`** standalone CLI toolkit — **now its own GitHub repo**
  `https://github.com/BrandoZhang/media-ai` (extracted from `packages/media-ai/`,
  which has been **deleted** from uni-agent).
- **`uni_agent/tools/media_gen/`** — thin tool *registrations* (schemas only)
  pointing at the media-ai console scripts. No python import of `media_ai`.
- **`uni_agent/reward/media_creation.py`** — cost-aware reward spec.
- **`uni_agent/workspace.py`** — per-run/session workspace isolation + GC helpers.
- **`apps/creation_gui/`** — decoupled web GUI (chat + tool calls + inline
  media + live cost) over an SSE event protocol.
- **`examples/creation_agent/`** — `demo.py` (standalone loop), `config.yaml`
  (UniAgentLoop/training), `mock_llm_server.py` (scripted OpenAI-compatible),
  `smoke_test.py` (no-LLM toolchain test), `README.md`.
- **`skills/<tool>/SKILL.md`** — one skill per generation tool (progressive
  disclosure; the agent `cat`s the relevant one on demand).

---

## 2. Repo map (where things live)

```
github.com/BrandoZhang/media-ai         ← standalone toolkit (separate repo)
  media_ai/mediakit.py                    backends (mock + Volcengine Ark) + HTTP + usage ledger
  media_ai/cli/*.py                       8 console scripts + `media-ai` dispatcher
  tests/                                  46 tests (mock backend, network-free Ark, CLI dispatch), CI

uni-agent (this repo)
  uni_agent/tools/media_gen/__init__.py   thin schema registrations (copy_to_remote=False system tools)
  uni_agent/reward/media_creation.py      reward = quality_proxy - cost_weight * total_tokens
  uni_agent/reward/registry.py            registers "media_creation"
  uni_agent/workspace.py                  resolve_media_workspace / compose_post_setup_cmd /
                                          should_gc_workspace / workspace_gc_command / gc_would_delete
  uni_agent/agent_loop.py                 UniAgentLoop.run wires workspace + GC (see §4.3–4.4)
  uni_agent/interaction/interaction.py    the loop; per-turn cost metering (_record_llm_cost/_record_tool_cost)
  uni_agent/interaction/model.py          _extract_openai_usage (prompt/completion/cached/reasoning)
  apps/creation_gui/{events,artifacts,session,server}.py + static/index.html + README.md
  examples/creation_agent/{demo.py,config.yaml,mock_llm_server.py,smoke_test.py,README.md}
  skills/{text2image,image2image,text2video,image2video,ref2video,concat_video}/SKILL.md
  tests/uni_agent/test_media_{workspace,creation_reward,gen_tools}.py
  tests/creation_gui/test_{events,artifacts}.py
```

`media-ai` is consumed as an **optional dependency**: `pip install ".[media]"`
(declared in root `pyproject.toml` → `project.optional-dependencies.media =
["media-ai @ git+https://github.com/BrandoZhang/media-ai"]`). uni-agent invokes
the tools as **console scripts on PATH**, never as a module — that's why the
split is clean.

---

## 3. The two integration paths (framework background)

- **Path 1 — `UniAgentLoop`** (`uni_agent/agent_loop.py`): bridges the
  `AgentInteraction` loop to verl's `AgentLoopBase` for parallel inference / RL
  training. This is where workspace isolation + GC + reward are wired.
- **Path 2 — `gateway/`**: a FastAPI/Ray OpenAI-compatible `/v1/chat/completions`
  server for black-box agent frameworks (Deep Agents / Claude Code / …). It
  serves the *model*; tool execution is the client's job in that protocol.

The GUI deliberately does **not** use Path 2 (see §4.5).

---

## 4. Key design decisions (with rationale)

### 4.1 media-ai is standalone, invoked as CLIs
Non-coding generation tasks don't need a sandbox for safety; they're IO + HTTP.
Making the tools a framework-agnostic CLI package means the *same* tools drop
into any agent sandbox (uni-agent, Claude Code, Codex, …). uni-agent registers
them as **system tools** (`copy_to_remote=False`): the runtime image must have
`media-ai` on PATH. **Ark API** (Volcengine) uses Bearer API-key auth; images
are sync (`/images/generations`), video is async (create → poll → cancel).

### 4.2 Cost as an evaluation metric (fine-grained)
- Per-turn metering in the loop accumulates `cost/llm/*` (prompt / completion /
  cached / reasoning / total tokens) and `cost/tool/<name>/*` into
  `rollout_cache["metrics"]` — generic: **any** tool that prints a JSON line
  with a `usage` block is metered (no allowlist).
- The **reward** derives cost from **this run's trajectory** (never a shared
  usage-ledger file) → concurrency-safe under parallel rollouts. It also
  **dedups async video tasks by task id** (a re-queried task isn't
  double-counted) and counts **footage by result shape, not tokens** (a
  zero-token real clip still contributes its seconds).
- `reward = quality_proxy - cost_weight * total_tokens`, clamped to [-1, 1].
  `quality_proxy` is a **placeholder** (did we produce a real film, ≥ a couple
  shots) — swap in a VLM/aesthetic judge for production.

### 4.3 Workspace isolation — keyed by a `workspace_key`
Concurrent rollouts on a **shared** FS (`local_native` / `host`) would collide
(the agent picks its own output names). Effective workspace =
`<MEDIA_WORKSPACE_BASE>/<workspace_key>`, where `workspace_key` =
`MEDIA_WORKSPACE_KEY` (a **stable** caller-supplied session id, for a resumable
interactive task that spans many model invocations) **else** the per-rollout
`run_id` (a one-shot training/eval trajectory — a fresh uuid is right). A run's
whole trajectory — however many turns, across a partial-rollout pause/resume —
stays one `run_id`, one workspace. Container backends isolate the FS per run for
free, so this derivation is a harmless no-op there. Opt-in via
`MEDIA_WORKSPACE_BASE`; a pinned static `MEDIA_WORKSPACE` is respected (back-compat).

### 4.4 Run-end GC — opt-in, sandbox-safe
`MEDIA_WORKSPACE_GC=reclaim` deletes `<base>/<run_id>` at run end (default off,
so demo/eval keep artifacts). **Critical subtlety:** artifacts live on the
*container's* FS for container backends and the *host* FS for local_native/host,
so deletion runs **through the env** (`rm -rf` inside the sandbox), never a
host-side `shutil.rmtree` (which could hit the wrong path for a container run).
It fires *after* the reward and *before* `env.close()`. Guards: never GC a
session-keyed dir; never GC a dir that contains the run's own `output_dir`
(`gc_would_delete`). Session-level + TTL/orphan reaping were deliberately left
to the harness (a crash never runs the finally block).

### 4.5 GUI is decoupled — option A′ (framework core untouched)
The browser depends only on a small HTTP+SSE event contract; it never imports
uni-agent. Only `session.py` touches uni-agent, and it drives the **public**
`AgentInteraction.step()` in its own loop (**A′** — no per-step hook was added
to the framework core; an earlier core-hook approach was reverted per the user's
"don't touch core" preference). Why not point the browser at the OpenAI gateway?
Because in that protocol tool execution is client-side, but the media tools must
run server-side — so the boundary is an **event stream**, not the model API.
Swap uni-agent for another framework by rewriting `session.py` alone.
Event types: `assistant` / `tool_call` / `tool_result{+artifact,+usage}` /
`cost` / `done` / `error`. Artifacts are opaque `/api/artifacts/<token>` URLs
**confined to the sessions root** (a tool observation can't make the server
serve arbitrary host files).

### 4.6 Env-var conventions
`ARK_*` for the Ark backend (`ARK_API_KEY`, `ARK_IMAGE_MODEL`, `ARK_VIDEO_MODEL`,
`ARK_POLL_TIMEOUT`, …) — standardized (no more `VOLC_`). `MEDIA_*` for the
toolkit/harness (`MEDIA_BACKEND`, `MEDIA_USAGE_LOG`, `MEDIA_WORKSPACE`,
`MEDIA_WORKSPACE_BASE`, `MEDIA_WORKSPACE_KEY`, `MEDIA_WORKSPACE_GC`).

---

## 5. Runtime facts (this environment)

- **No GPU** (≈4 CPU / 15 GB). A real tool-calling vLLM is impractical → the
  demo/GUI auto-start a **scripted mock LLM** (`mock_llm_server.py`) that plays
  a fixed storyboard regardless of the brief. Set `BASE_URL` for a real model.
- Media tools: mock backend = Pillow images + bundled ffmpeg (`imageio-ffmpeg`),
  deterministic, offline; `volc` backend = real Ark API.
- Installed for local runs: `swe-rex openai loguru pydantic pydantic_settings
  orjson regex numpy pytest ruff` + `pip install ".[media]"` (media-ai) + its
  Pillow/imageio-ffmpeg deps.

---

## 6. How to run / test / verify

```bash
# unit tests (fast, no loop, no GPU)
python -m pytest tests/uni_agent/test_media_workspace.py \
  tests/uni_agent/test_media_creation_reward.py \
  tests/uni_agent/test_media_gen_tools.py tests/creation_gui/ -q

# no-LLM toolchain smoke (needs media-ai installed)
python examples/creation_agent/smoke_test.py

# full agent loop offline (mock LLM + mock media)
python examples/creation_agent/demo.py

# the GUI (open http://127.0.0.1:8770)
python -m apps.creation_gui.server --port 8770

# lint (repo excludes verl/ and uni_agent/tools/ from ruff+compileall)
ruff check <changed files>
```

Real generation: set `MEDIA_BACKEND=volc ARK_API_KEY=...` (+ optional
`BASE_URL`/`MODEL_NAME` for a real LLM). Ark model IDs are account-specific
(`ARK_IMAGE_MODEL` / `ARK_VIDEO_MODEL` or per-call `--model`); list at
https://www.volcengine.com/docs/82379/1330310 .

---

## 7. Next steps / open work (prioritized)

**Needs the media-ai repo (the reason for a new session with repo access):**
1. **Publish `media-ai` to PyPI** + add a GitHub release workflow (tag → build →
   publish). Then **flip every `git+https://github.com/BrandoZhang/media-ai`
   reference to `pip install media-ai`** here (root `pyproject.toml` extra,
   `examples/creation_agent` README/demo/smoke_test/config, `uni_agent/tools/
   media_gen` docstrings). Grep: `git+https://github.com/BrandoZhang/media-ai`.
2. Consider a `media-ai` version pin in the extra once released.

**In uni-agent (no external repo needed):**
3. **GUI real-volc mid-flight cancel**: today cancel is cooperative (between
   steps). Wire the `video_task --op cancel` lever so a queued/rendering Volc
   video is actually interrupted (cost control). `mediakit._poll_video` already
   cancels on SIGTERM/SIGINT + its own timeout; the GUI just needs to reach it.
4. **Mock LLM that understands the brief** (currently ignores it and plays a
   fixed storyboard) — makes the offline GUI feel real for arbitrary briefs.
5. **Training-observation dashboard**: extend `dashboard/` to surface each run's
   `metrics` (`cost/*`, `reward_score`) from `interaction_result.json` — a
   cross-run cost/reward comparison view (the existing dashboard is a rollout/
   log monitor, not a metrics comparator).
6. **Biggest quality lever — multimodal observation feedback**: feed generated
   images/frames back into a VLM in the loop so the agent can *see* and critique
   its output (vs. flying blind on text). This is the main future enhancement.

**Design decisions intentionally deferred (revisit, don't silently flip):**
7. Reward shaping: `quality (≤1) - cost_weight*tokens` inverts past ~5 clips
   (a good big film can score below doing nothing). Left as a tuning knob;
   a *saturating* cost term would bound the penalty — a real design change.
8. Retry policy: POST create-task is **not** retried on `URLError` (a
   connection-level failure could be a pre- or post-send failure; retrying risks
   double-submitting a billed task). Kept conservative by design.

---

## 8. Process constraints (so a new session doesn't trip)

- **Branch**: develop on `claude/laughing-lamport-r3ws5m`; never push elsewhere
  without permission. If its PR was already merged, restart the branch from the
  latest default branch for follow-up work (don't stack on merged history).
- **No PR** unless the user explicitly asks.
- **Commit trailers** (every commit): `Co-Authored-By: Claude Opus 4.8
  <noreply@anthropic.com>` and `Claude-Session: <session url>`.
- **Never** put the raw model identifier (the `claude-...` id string) in any
  committed artifact — chat only.
- **Secrets**: `ARK_API_KEY` etc. are secrets — keep them commented in configs,
  never hardcode.
- **GitHub scope**: *this* session's GitHub tools were restricted to
  `brandozhang/uni-agent` only (couldn't create/add other repos) — hence the
  media-ai repo was handed off as a git bundle for the user to push. A new
  session with broader scope can operate on `BrandoZhang/media-ai` directly.

---

## 9. Known gotchas

- **Stale tool shadowing**: an old `~/.uni-agent/bin/text2image` (a former
  `copy_to_remote=True` copy) can shadow the media-ai console script on PATH and
  cause `ModuleNotFoundError: uni_agent`. Fresh sandboxes are fine; if it bites,
  `rm -f ~/.uni-agent/bin/{text2image,...}`.
- **Container images must preinstall media-ai** (system tool on PATH) — the
  `[media]` extra is the standard install entry, but each backend image
  (modal/vefaas/local) still has to bake it in out of band.
- **`_poll_video` cancel can't catch SIGKILL** — only SIGTERM/SIGINT + its own
  timeout. For real video prefer `--wait false` + `video_task`, or set
  `action_timeout >= ARK_POLL_TIMEOUT`.
- **concat audio is all-or-nothing**: `concat_video` preserves audio only when
  *every* input clip has an audio track; mixed inputs fall back to video-only
  (silence-padding was out of scope).
- **Mock LLM ignores the brief** (fixed storyboard) — see next-step #4.
