#!/usr/bin/env python3
"""No-LLM smoke test: exercise the media-gen CLIs end-to-end offline.

Runs the storyboard pipeline directly (no model, no uni_agent loop) so the
toolchain can be verified anywhere Pillow + ffmpeg are available:

  text2image -> image2image -> image2video -> text2video -> ref2video
             -> concat_video -> media_usage

Asserts each artifact exists and is non-empty, and that the usage ledger
aggregates. Exits non-zero on any failure.

    python examples/creation_agent/smoke_test.py [--workdir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "uni_agent" / "tools" / "media_gen" / "cli"


def run(name: str, *args: str) -> dict:
    proc = subprocess.run([sys.executable, str(CLI / name), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] {name}: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    out = proc.stdout.strip()
    print(f"[ok] {name}: {out[:140]}")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def assert_file(path: str, label: str) -> None:
    p = Path(path)
    if not (p.is_file() and p.stat().st_size > 0):
        print(f"[FAIL] {label}: missing/empty {path}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    tmp = args.workdir or tempfile.mkdtemp(prefix="creation_smoke_")
    w = Path(tmp)
    w.mkdir(parents=True, exist_ok=True)
    os.environ["MEDIA_USAGE_LOG"] = str(w / "usage.jsonl")
    os.environ.setdefault("MEDIA_BACKEND", "mock")
    print(f"workdir: {w}\nbackend: {os.environ['MEDIA_BACKEND']}\n")

    r = run(
        "text2image",
        "--prompt",
        "silver-suited astronaut on a red dune, cinematic",
        "--output",
        str(w / "ref.png"),
        "--seed",
        "7",
    )
    assert_file(r["path"], "text2image")

    r = run(
        "image2image",
        "--images",
        json.dumps([str(w / "ref.png")]),
        "--prompt",
        "same astronaut, low angle",
        "--output",
        str(w / "ref2.png"),
        "--seed",
        "7",
    )
    assert_file(r["path"], "image2image")

    r = run(
        "image2video",
        "--first_frame",
        str(w / "ref.png"),
        "--prompt",
        "astronaut turns to camera",
        "--output",
        str(w / "shot1.mp4"),
        "--seconds",
        "3",
        "--resolution",
        "480p",
        "--return_last_frame",
        "true",
    )
    assert_file(r["path"], "image2video")
    assert r["extra_paths"], "image2video should return a last frame"

    r = run(
        "text2video",
        "--prompt",
        "twin suns setting over the dunes",
        "--output",
        str(w / "shot2.mp4"),
        "--seconds",
        "3",
        "--resolution",
        "480p",
    )
    assert_file(r["path"], "text2video")

    r = run(
        "ref2video",
        "--images",
        json.dumps([str(w / "ref.png"), str(w / "ref2.png")]),
        "--prompt",
        "two astronauts meet",
        "--output",
        str(w / "shot3.mp4"),
        "--seconds",
        "3",
    )
    assert_file(r["path"], "ref2video")

    r = run(
        "concat_video",
        "--inputs",
        json.dumps([str(w / "shot1.mp4"), str(w / "shot2.mp4"), str(w / "shot3.mp4")]),
        "--output",
        str(w / "final.mp4"),
    )
    assert_file(r["path"], "concat_video")

    r = run("media_usage")
    totals = r["totals"]
    assert totals["total_tokens"] > 0, "usage ledger should have accumulated tokens"
    assert totals["images_generated"] >= 2 and totals["video_seconds"] >= 9

    print(f"\n[PASS] full pipeline. final={w / 'final.mp4'} ({(w / 'final.mp4').stat().st_size} bytes)")
    print(
        f"[PASS] cost: {totals['total_tokens']} tokens across {totals['calls']} generation calls -> {totals['by_tool']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
