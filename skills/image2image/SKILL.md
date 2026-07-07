---
name: image2image
description: How to use the image2image tool — generate a new image from one or more reference images plus a text prompt. Read this when you need a consistent variant of an existing image (same character/style in a new pose/angle/scene), or when you want to fuse several reference images into one.
---

# image2image — reference image(s) + text → image

Derive a new image from 1..N reference images and a prompt.

## Command
```
image2image --images '["ref.png"]' --prompt "<description>" --output <path.png> [--strength 0.6] [--max_images 1] [--seed N] [--backend mock|volc]
```
`--images` takes a JSON array (or one/more plain paths). Prints a JSON line
with the artifact `path` and a `usage` block.

## When to use it
- You already have a reference (e.g. one made by `text2image`, or given by
  the user) and want a **consistent variant** — same character/style, new
  pose, angle, expression, or setting.
- **Multi-image fusion**: pass several references to combine a character +
  a setting + a style into one image.
- To produce a fresh still that you then animate with `image2video` while
  keeping the look locked.

## Options that matter
- `--strength` (0-1): how much to follow the prompt vs. preserve the
  reference. Lower keeps the reference closer; higher lets the prompt reshape it.
- `--max_images N`: request a related group only if you need it.
- `--seed`: reuse the same seed to keep variants visually matched.

## Consistency
Reuse the *same* reference file wherever a subject must stay identical —
don't regenerate the character from scratch each time, or continuity breaks.

## Cost
Scales with pixels/images; check `usage`. Prefer the fewest, smallest images
that meet the brief.
