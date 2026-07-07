---
name: text2video
description: How to use the text2video tool — generate a short video clip from a text prompt only (no input image). Read this for establishing/scenery shots, abstract motion, or any shot that does not need to match a specific reference image.
---

# text2video — text → video clip

Generate one short clip from a text prompt. No input image.

## Command
```
text2video --prompt "<action>" --output <shot.mp4> [--seconds 5] [--resolution 480p] [--ratio 16:9] [--seed N] [--camera_fixed false] [--watermark false] [--generate_audio false] [--backend mock|volc]
```
The real (volc) backend submits an async task and blocks until the clip is
ready. Prints a JSON line with the artifact `path` and a `usage` block.

## When to use it
- Establishing / scenery / B-roll shots with **no recurring subject** that
  must match a specific image.
- If a shot must stay consistent with a character/style you've locked, use
  `image2video --first_frame <ref>` instead.

## Prompt tips
Describe the **action and camera motion**, not just a static scene:
subject + what it does + camera move (push-in, pan, aerial) + mood. Keep it
focused; overlong prompts scatter the result.

## Options that matter
- `--seconds`: clip length (keep short, 3-5s, for a tiny film).
- `--resolution` (480p|720p|1080p): **cost scales steeply with resolution.**
  Default to 480p unless the brief demands more.
- `--ratio`: 16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9.
- `--camera_fixed`: lock the camera. `--generate_audio`: synced audio
  (Seedance 2.0/1.5). `--seed`: reproducibility.

## Cost
Video is the most expensive operation; `completion_tokens` scales with
resolution × duration. Cost is an evaluation metric — use the lowest
resolution and shortest duration that meet the brief.

## Async (long videos)
Real video generation can take minutes. To avoid blocking, submit with `--wait false` (volc backend): it returns a `task_id` immediately. Then poll with `video_task --op query --id <task_id> --output <path>`, which downloads the clip once the task succeeds. Cancel a queued task with `video_task --op cancel --id <task_id>` to save cost. (The mock backend is synchronous and ignores `--wait`.)

For **multiple shots**, you can emit several video tool calls in a single turn (the harness runs each and returns all results) rather than doing one shot per turn.
