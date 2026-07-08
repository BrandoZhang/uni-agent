---
name: image2video
description: How to use the image2video tool — generate a video clip whose first frame is a given image (optionally also a last frame). Read this whenever a shot must stay visually consistent with a reference image; it is the main tool for character/style continuity across shots, for first+last-frame interpolation, and for chaining continuous shots.
---

# image2video — first-frame (+ optional last-frame) image → video

Animate from a locked starting image. This is the key tool for **cross-shot
consistency**.

## Command
```
image2video --first_frame <ref.png> --prompt "<motion>" --output <shot.mp4> [--last_frame <img.png>] [--seconds 5] [--resolution 480p] [--ratio adaptive] [--return_last_frame false] [--seed N] [--model <id>] [--backend mock|volc]
```
The real (volc) backend submits an async task and blocks until ready. Prints
a JSON line with the artifact `path`, `usage`, and any `extra_paths` (the
returned last frame).

## When to use it
- **Any shot featuring a recurring subject** — pass the locked reference
  (from `text2image`/`image2image`, or a user asset) as `--first_frame` so
  the shot starts from the exact look. Prefer this over `text2video` for
  continuity.
- **First+last-frame interpolation**: give `--last_frame` too to control
  where the shot ends.
- **Chaining continuous shots**: set `--return_last_frame true` to get the
  clip's final frame back, then feed it as the next shot's `--first_frame`
  for a seamless sequence.

## Consistency workflow
1. Make/keep one reference image per recurring subject.
2. Use it as `--first_frame` for every shot that features that subject.
3. Reuse the *same* file — don't regenerate the subject per shot.

## Options that matter
- `--ratio adaptive` matches the first frame's aspect ratio (recommended).
- `--resolution`: cost scales steeply; default 480p.
- `--seconds`: keep short (3-5s).

## Cost
Video is expensive; `completion_tokens` scales with resolution × duration.
Cost is an evaluation metric — minimize resolution/duration and avoid
redundant regenerations. Check `usage`.

## Async (long videos)
Real video generation can take minutes. To avoid blocking, submit with `--wait false` (volc backend): it returns a `task_id` immediately. Then poll with `video_task --op query --id <task_id> --output <path>`, which downloads the clip once the task succeeds. Cancel a queued task with `video_task --op cancel --id <task_id>` to save cost. (The mock backend is synchronous and ignores `--wait`.)

For **multiple shots**, you can emit several video tool calls in a single turn (the harness runs each and returns all results) rather than doing one shot per turn.

## Model
`--model` selects the Ark video Model ID (optional). Default:
`doubao-seedance-2-0-260128` (override globally with `$ARK_VIDEO_MODEL`).
Other families: `doubao-seedance-1-5-pro-*`, `doubao-seedance-1-0-pro-*`. A
model must be **enabled for your account**; see the full Model ID list at
https://www.volcengine.com/docs/82379/1330310 (or the Ark console model list).
