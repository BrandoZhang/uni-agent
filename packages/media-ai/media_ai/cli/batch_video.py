"""batch_video: generate many video shots concurrently, then join.

Fan out several video jobs at once with a **bounded concurrency** cap (so you
don't trip the provider's rate/concurrency limits), then wait for all of them.
Turns an N-shot storyboard from N sequential multi-minute waits into roughly
one — the big win of async generation.

Parameters:
  --jobs (string, required): a JSON array of job objects. Each job:
      {"tool": "text2video"|"image2video"|"ref2video", "output": "shotN.mp4",
       "prompt": "...", "seconds": 5, "resolution": "480p", ...plus that
       tool's own params, e.g. "first_frame" for image2video, "images" for
       ref2video}.
  --max_concurrent (int, optional): max in-flight generations. Default 4
      (or $VOLC_MAX_CONCURRENT_VIDEO). Keep it modest to avoid 429s; the
      backend already retries transient rate limits with backoff.
  --backend (string, optional): mock (default) or volc.

Failures are isolated: one bad job is reported but the rest still complete
(partial success). Prints a JSON summary with per-job results + total cost.

Example:
  batch_video --max_concurrent 3 --jobs '[
    {"tool":"image2video","first_frame":"ref.png","prompt":"turn","output":"s1.mp4","seconds":3},
    {"tool":"text2video","prompt":"wide desert dusk","output":"s2.mp4","seconds":3}
  ]'
"""

import argparse
import json
import sys

from media_ai import mediakit


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate many video shots concurrently, then join.")
    ap.add_argument("--jobs", required=True, help="JSON array of job objects.")
    ap.add_argument("--max_concurrent", type=int, default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    try:
        jobs = json.loads(args.jobs)
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("--jobs must be a non-empty JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: invalid --jobs: {e}", file=sys.stderr)
        return 1

    try:
        summary = mediakit.batch_videos(jobs, backend=args.backend, max_concurrent=args.max_concurrent)
    except mediakit.MediaError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    # Non-zero exit only if *every* job failed; partial success is still a success.
    return 0 if summary["succeeded"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
