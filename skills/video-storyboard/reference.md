# video-storyboard — tool reference

All generation tools print a one-line JSON result to stdout including the
artifact `path` and a `usage` block (token cost). On failure they print
`Error: ...` to stderr and exit non-zero.

## Generation tools

### text2image
Generate an image (or a related group) from text. Use for reference assets.
```
text2image --prompt "<desc>" --output <path.png> [--width 768] [--height 432] [--max_images 1] [--seed N] [--backend mock|volc]
```

### image2image
Generate image(s) from one or more reference images + prompt (locks a look
or fuses references).
```
image2image --images '["ref.png"]' --prompt "<desc>" --output <path.png> [--strength 0.6] [--max_images 1] [--seed N] [--backend mock|volc]
```

### text2video
Generate a clip from text. Use for scenery/establishing shots.
```
text2video --prompt "<action>" --output <shot.mp4> [--seconds 5] [--resolution 480p] [--ratio 16:9] [--seed N] [--camera_fixed false] [--watermark false] [--generate_audio false] [--backend mock|volc]
```

### image2video
Generate a clip from a first frame (+ optional last frame). **Preferred for
any shot with a recurring subject** — this keeps shots consistent.
```
image2video --first_frame <ref.png> --prompt "<motion>" --output <shot.mp4> [--last_frame <img.png>] [--seconds 5] [--resolution 480p] [--ratio adaptive] [--return_last_frame false] [--seed N] [--backend mock|volc]
```
Set `--return_last_frame true` to get the clip's final frame back, then feed
it as the next shot's `--first_frame` for a continuous sequence.

### ref2video
Multimodal-reference generation: mix reference images (0-9), videos (0-3),
audio (0-3) + optional prompt into one clip (Seedance 2.0). At least one
image or video is required.
```
ref2video --images '["a.png","b.png"]' --prompt "<desc>" --output <shot.mp4> [--videos '[...]'] [--audios '[...]'] [--seconds 5] [--resolution 480p] [--backend mock|volc]
```

### concat_video
Join clips, in order, into the final film.
```
concat_video --inputs '["shot1.mp4","shot2.mp4","shot3.mp4"]' --output final.mp4 [--width 768] [--height 432]
```

## Utility tools

### video_task
Query or cancel an async video task by id (volc backend). Cancelling a
queued task cuts cost.
```
video_task --op query|cancel --id <task_id>
```

### media_usage
Report accumulated token cost from the usage ledger. Call it before you
finish to summarize the run's cost.
```
media_usage
```

## shots.json schema

Write your plan here before generating, then execute it shot by shot.

```json
{
  "title": "short human title",
  "workspace": "/abs/path/to/workspace",
  "reference_assets": [
    {"id": "hero", "path": "workspace/ref_hero.png", "desc": "the recurring subject + style"}
  ],
  "shots": [
    {"id": 1, "description": "the motion", "camera": "slow push-in", "seconds": 4,
     "reference": "workspace/ref_hero.png", "tool": "image2video", "output": "workspace/shot1.mp4"},
    {"id": 2, "description": "establishing scenery", "camera": "static wide", "seconds": 3,
     "reference": null, "tool": "text2video", "output": "workspace/shot2.mp4"}
  ],
  "final": "workspace/final.mp4"
}
```

## Consistency & cost checklist
- One reference file per recurring subject; reuse it as `--first_frame` across shots.
- Same `--seed` for shots that should feel visually matched.
- **Cost is an evaluation metric — minimize it.** Keep clips short (3-5s),
  use the lowest resolution that meets the brief (`480p` by default), avoid
  regenerating assets you already have, and don't request group images you
  don't need. Call `media_usage` at the end and report the total.
