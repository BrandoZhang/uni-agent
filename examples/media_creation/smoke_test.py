#!/usr/bin/env python3
"""No-LLM smoke test: exercise the media-ai CLI end-to-end offline.

Runs the storyboard pipeline directly (no model, no agent loop) via the
standalone media-ai package's unified ``media-ai <group> <op>`` CLI, so the
toolchain can be verified anywhere it is installed (``pip install -e
../media-ai``; needs Pillow + ffmpeg):

  image generate -> image edit -> video generate (first-frame) ->
  video generate (text) -> concat -> usage

Asserts each artifact exists and is non-empty, and that the usage ledger
aggregates. Exits non-zero on any failure.

    python examples/media_creation/smoke_test.py [--workdir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*argv: str) -> dict:
    """Invoke ``media-ai <argv...>`` (via the module dispatcher) and parse its JSON line."""
    proc = subprocess.run([sys.executable, "-m", "media_ai", *argv], capture_output=True, text=True)
    label = " ".join(argv[:2] if len(argv) > 1 and not argv[1].startswith("-") else argv[:1])
    if proc.returncode != 0:
        print(f"[FAIL] {label} (exit {proc.returncode}): {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    out = proc.stdout.strip()
    print(f"[ok] {label}: {out[:140]}")
    try:
        return json.loads(out.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
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

    w = Path(args.workdir or tempfile.mkdtemp(prefix="media_smoke_"))
    w.mkdir(parents=True, exist_ok=True)
    os.environ["MEDIA_USAGE_LOG"] = str(w / "usage.jsonl")
    os.environ.setdefault("MEDIA_PROVIDER", "mock")  # offline, credential-free default
    print(f"workdir: {w}\nprovider: {os.environ['MEDIA_PROVIDER']}\n")

    r = run("image", "generate", "--prompt", "silver-suited astronaut on a red dune, cinematic",
            "--output", str(w / "ref.png"), "--seed", "7")
    assert_file(r["path"], "image generate")

    r = run("image", "edit", "--reference", str(w / "ref.png"), "--prompt", "same astronaut, low angle",
            "--output", str(w / "ref2.png"), "--seed", "7")
    assert_file(r["path"], "image edit")

    r = run("video", "generate", "--first-frame", str(w / "ref.png"), "--prompt", "astronaut turns to camera",
            "--output", str(w / "shot1.mp4"), "--seconds", "3", "--resolution", "480p", "--return-last-frame", "true")
    assert_file(r["path"], "video generate (first-frame)")
    assert r["extra_paths"], "video generate --return-last-frame should return a last frame"

    r = run("video", "generate", "--prompt", "twin suns setting over the dunes",
            "--output", str(w / "shot2.mp4"), "--seconds", "3", "--resolution", "480p")
    assert_file(r["path"], "video generate (text)")

    r = run("concat", "--input", str(w / "shot1.mp4"), "--input", str(w / "shot2.mp4"),
            "--output", str(w / "final.mp4"))
    assert_file(r["path"], "concat")

    r = run("usage")
    totals = r["totals"]
    assert totals["total_tokens"] > 0, "usage ledger should have accumulated tokens"
    assert totals["images_generated"] >= 2 and totals["video_seconds"] >= 6, totals

    print(f"\n[PASS] full pipeline. final={w / 'final.mp4'} ({(w / 'final.mp4').stat().st_size} bytes)")
    print(f"[PASS] cost: {totals['total_tokens']} tokens across {totals['calls']} calls -> {totals['by_tool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
