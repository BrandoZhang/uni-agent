---
name: ref2video
description: How to use the ref2video tool — multimodal-reference video generation (Seedance 2.0). Read this when a shot should be driven by a mix of reference images (0-9), reference videos (0-3) and/or reference audio (0-3) plus an optional prompt — for asset-driven consistency beyond a single first frame, for video editing/extension, or for audio-synced generation.
---

# ref2video — multimodal references → video clip

Generate one clip from any mix of reference images, videos, and audio, plus
an optional prompt. At least one reference image or video is required.

## Command
```
ref2video --images '["a.png","b.png"]' --prompt "<description>" --output <shot.mp4> [--videos '[...]'] [--audios '[...]'] [--seconds 5] [--resolution 480p] [--ratio adaptive] [--seed N] [--watermark false] [--generate_audio false] [--backend mock|volc]
```
`--images/--videos/--audios` each take a JSON array (or one/more paths/URLs).
The real (volc) backend submits an async task and blocks until ready.

## When to use it
- You need **more than a single first frame** to drive consistency — e.g.
  multiple reference images of a character/props/setting.
- **Video editing / extension**: pass a reference video to edit or continue it.
- **Audio-synced** generation: pass reference audio (must accompany at least
  one reference image or video — audio alone is not allowed).
- For a simple "start from this one image" shot, prefer `image2video` instead.

## References & roles
- images → `reference_image` (0-9), videos → `reference_video` (0-3),
  audio → `reference_audio` (0-3).
- **Reference videos/audio should generally be public URLs or `asset://`
  IDs.** Large local files are inlined as base64, which the API may reject.
  Local reference *images* are fine as base64.

## Cost
Video is expensive and multimodal inputs add tokens; check `usage`. Use the
lowest resolution / shortest duration that meets the brief.

## Async (long videos)
Real video generation can take minutes. To avoid blocking, submit with `--wait false` (volc backend): it returns a `task_id` immediately. Then poll with `video_task --op query --id <task_id> --output <path>`, which downloads the clip once the task succeeds. Cancel a queued task with `video_task --op cancel --id <task_id>` to save cost. (The mock backend is synchronous and ignores `--wait`.)
