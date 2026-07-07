---
name: concat_video
description: How to use the concat_video tool — join several video clips, in order, into one final film. Read this when you have generated individual shots and need to assemble the deliverable.
---

# concat_video — join clips → final film

Concatenate per-shot clips into one video. Local ffmpeg (no API, no token cost).

## Command
```
concat_video --inputs '["shot1.mp4","shot2.mp4","shot3.mp4"]' --output final.mp4 [--width 768] [--height 432]
```
`--inputs` takes a JSON array of clip paths **in story order**. Prints a JSON
line with the final `path` and clip count.

## When to use it
- As the final assembly step, once every shot clip exists and you've verified
  each one's result JSON has `"ok": true` and a real `path`.

## Notes
- Order matters — list clips in the order they should play.
- Clips are re-encoded and normalized (size/fps) so differing inputs join
  cleanly; keeping shots at the same resolution avoids letterboxing.
- This is a local operation and does not spend generation tokens.
