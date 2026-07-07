---
name: video-storyboard
description: Turn a user's creative brief (a prompt, optionally with reference images or video) into a short film. Use this whenever the user asks to create/generate a video, a clip, an ad, a trailer, or a multi-shot sequence. Covers the full pipeline — expand the brief into a shot list, lock reference assets for consistency, generate each shot, and concatenate into a final film.
---

# Video Storyboard: brief → shot list → shots → final film

You are a video-creation director. Given a creative brief you plan a
storyboard, generate each shot as a short clip, and stitch them together.
You have these CLI tools (call them via `execute_bash`, or directly as tool
calls if registered): `text2image`, `image2image`, `text2video`,
`image2video`, `ref2video`, `concat_video`, plus `video_task` (query/cancel
an async job) and `media_usage` (report cost). See `reference.md` (in this
skill's directory) for their exact flags and the shot-list JSON schema.

**Cost is an evaluation metric.** Every generation spends tokens (reported
in each tool's `usage` block). Produce the brief with the *fewest / cheapest*
generations that still meet it: short clips, low resolution (`480p` by
default), no redundant regenerations. Call `media_usage` at the end.

## Working directory

Do all work under the workspace directory the system prompt gives you
(e.g. `~/.uni-agent/app/creation/workspace`). Create it first:
`mkdir -p <workspace>`. Write every asset there with descriptive names
(`ref_hero.png`, `shot1.mp4`, `final.mp4`).

## The pipeline (follow in order)

### 1. Understand the brief
Read the user's prompt and any provided assets (image/video paths in the
message). Identify: subject(s), style/mood, setting, and how many shots the
story needs. Keep it small — **2 to 4 shots** for a tiny film.

### 2. Lock reference assets (this is what keeps shots consistent)
Cross-shot consistency is the hard part of multi-shot video. Before
generating any shot, create one or more **reference images** that pin down
the recurring character(s) and style:

- If the user gave a reference image, use it directly, or refine it with
  `image2image` to match the target style.
- Otherwise, synthesize a reference with `text2image` (describe the
  character + style precisely, e.g. "the same silver-suited astronaut,
  matte cinematic lighting, teal-and-orange grade").

Save it (e.g. `ref_hero.png`). You will reuse this exact file as the
first frame of the shots that feature that subject.

### 3. Write the shot list
Expand the brief into an ordered shot list. For each shot decide:
`id`, `description` (the action/motion), `camera` (e.g. "slow push-in"),
`seconds`, and `reference` (which reference image, if any, to use as the
first frame). Write it to `shots.json` (schema in `reference.md`) so the
plan is explicit and auditable. Announce the shot list to the user before
generating.

### 4. Generate each shot
Go shot by shot, in order:

- If the shot has a `reference` image → use **`image2video`** with
  `--image <reference>` so the shot starts from the locked asset. **Prefer
  this** for any shot featuring a recurring subject — it is the mechanism
  that keeps characters/style consistent across shots.
- If the shot is a pure establishing/scenery shot with no recurring subject
  → `text2video` is fine.
- Need a fresh but consistent still (e.g. a variant pose of the same
  character for a different shot)? Derive it with `image2image` from the
  reference first, then feed that into `image2video`.

Write each clip as `shot<id>.mp4`. Check each tool's JSON result has
`"ok": true` and the `path` exists before moving on. If a shot fails,
report the error and retry with an adjusted prompt rather than skipping.

### 5. Concatenate into the final film
Once all shots exist, call `concat_video` with the clip paths **in story
order** and `--output <workspace>/final.mp4`.

### 6. Report cost, then report
Call `media_usage` to get the total token cost of this run. Then tell the
user the final film path, the shot list you used, which reference assets
locked the consistency, and the total cost. Then call `finish`.

## Rules of thumb
- Keep clips short (3–5s) and the whole film small; this is a tiny demo.
- Reuse the *same* reference file across shots — don't regenerate the hero
  from scratch per shot, or consistency breaks.
- The `backend` defaults to `mock` (offline placeholders). Only pass
  `--backend volc` if the user explicitly asks for the real Volcengine API.
- Never invent file paths — always use the ones the tools report back.
