---
name: text2image
description: How to use the text2image tool — generate a still image (or a related group of images) from a text prompt only, with no input image. Read this when the task calls for creating new key art, a reference/character asset, a style frame, a poster, or a storyboard still from a description.
---

# text2image — text → image

Generate one image (or a related group) from a text prompt. No input image.

## Command
```
text2image --prompt "<description>" --output <path.png> [--width 768] [--height 432] [--max_images 1] [--seed N] [--model <id>] [--backend mock|volc]
```
Prints one JSON line with the artifact `path` and a `usage` (token cost) block.

## Prompt tips
Describe, in one focused prompt (≤ ~300 Chinese / ~600 English words):
- **subject** (who/what, appearance details),
- **style / medium** (photoreal, anime, oil painting, ...),
- **lighting & mood**, **camera** (angle, lens, shot size),
- **color grade** (e.g. "teal-and-orange cinematic grade").
Too many competing details dilute the result — keep the most important ones.

## When to use it
- You need a brand-new still and have no reference image. (If you already
  have a reference image and want a consistent variant, use `image2image`.)
- To mint a **reference/key asset** you will reuse to keep later shots
  consistent (then feed it to `image2video --first_frame`).

## Options that matter
- `--max_images N` (N>1): generate a *related group* (saved as `output`,
  `output_2`, ...). Only request as many as you actually need — each costs.
- `--width/--height`: aspect ratio and size. Larger = more tokens.
- `--seed`: fix for reproducibility / to match a look across images.

## Cost
Image cost scales with pixels (`output_tokens ≈ images · ⌊w·h/256⌋`). Cost is
an evaluation metric — use the smallest size and fewest images that meet the
brief. Check the `usage` in the result.

## Model
`--model` selects the Ark image Model ID (optional). Default:
`doubao-seedream-4-5-251128` (override globally with `$ARK_IMAGE_MODEL`).
Examples: `doubao-seedream-5-0-260128`, `doubao-seedream-4-5-251128`,
`doubao-seedream-4-0-250828`. A model must be **enabled for your account**;
see the full Model ID list at
https://www.volcengine.com/docs/82379/1330310 (or the Ark console model list).
