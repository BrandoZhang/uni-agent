---
name: batch_video
description: How to use the batch_video tool — generate several video shots concurrently (with a bounded concurrency cap + automatic retry on rate limits), then wait for them all. Read this whenever you need more than one video shot; it is the right way to produce a multi-shot sequence, turning N sequential multi-minute waits into roughly one.
---

# batch_video — fan out many shots, then join

Generate multiple video clips **at once** instead of one-by-one. Real video
generation takes minutes per shot, so running shots sequentially is N×
slower; batching runs them concurrently (bounded) and joins.

## Command
```
batch_video --jobs '<json array>' [--max_concurrent 4] [--backend mock|volc]
```

`--jobs` is a JSON array of job objects. Each job picks a video tool and
carries that tool's params, plus an `output` path:

```json
[
  {"tool": "image2video", "first_frame": "workspace/ref_hero.png",
   "prompt": "the hero turns to camera", "output": "workspace/shot1.mp4",
   "seconds": 3, "resolution": "480p"},
  {"tool": "text2video", "prompt": "wide desert at dusk",
   "output": "workspace/shot2.mp4", "seconds": 3, "resolution": "480p"},
  {"tool": "ref2video", "images": ["workspace/a.png","workspace/b.png"],
   "prompt": "the two meet", "output": "workspace/shot3.mp4", "seconds": 3}
]
```

Supported `tool` values: `text2video`, `image2video`, `ref2video`. Each job
takes the same params as that tool's own skill.

## Concurrency & rate limits
- `--max_concurrent` (default 4) caps how many generations run at once, so
  you don't trip the provider's rate/concurrency limit. Keep it modest.
- The backend automatically **retries transient rate limits (HTTP 429) and
  5xx with exponential backoff**, so brief bursts recover on their own.

## Robustness
- Jobs are isolated: if one fails it's reported in that job's result with an
  `error`, and the **other jobs still complete** (partial success). Inspect
  the per-job `ok` flags and retry only the failed ones.

## Output
A JSON summary: `succeeded`/`failed` counts, an aggregate `usage`
(total tokens), and a `results` array with each job's `ok`, `path`, and
`usage`. After it returns, `concat_video` the successful shots in order.

## Typical flow
1. Prepare reference assets (`text2image` / `image2image`) if shots need them.
2. Build the job list (one entry per shot) and call `batch_video`.
3. Check the summary; regenerate any failed shots.
4. `concat_video` the shot outputs, in order, into the final film.
