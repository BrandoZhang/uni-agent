# ruff: noqa: E501
"""Preprocess image-generation *intent* prompts into Parquet for uni-agent training.

The "dataset" for a generation agent is just a list of user intents (like DanceGRPO's
``data/prompt.txt``); the images themselves are produced by the frozen ``generate_image``
tool during rollout and scored by the ``image_agent`` reward. Each row is emitted in the
schema UniAgentLoop consumes:

  - ``prompt``         : [system, user] chat messages
  - ``agent_name``     : "image_agent"  (MUST match examples/image_agent/agent_config.yaml `name`)
  - ``extra_info.tools_kwargs.reward.ground_truth`` : optional per-sample reward metadata
                          (here, ``{"keywords": [...]}`` for the offline alignment slot)
  - ``reward_model``   : mirror of the ground truth (rule-style)

Usage:
  python examples/data_preprocess/image_agent.py --local_save_dir /tmp/image_agent_data
  # or bring your own newline-separated prompts (one intent per line):
  python examples/data_preprocess/image_agent.py --prompts_file my_prompts.txt --local_save_dir /tmp/image_agent_data
"""

import argparse
import logging
import os
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_CONTENT = (
    "You are an image-generation agent. Turn the user's request into a vivid, detailed image.\n"
    "Every assistant turn MUST contain EXACTLY ONE tool call.\n"
    "Workflow:\n"
    "1. Write a detailed prompt (subject, style, composition, lighting, colors) and call `generate_image`.\n"
    "2. Inspect the image you get back; if it can be improved, refine the prompt and call `generate_image` again.\n"
    "3. When satisfied, call `finish` with a short description of what you delivered.\n"
)

# A small built-in set of intents with keyword ground truth for the offline alignment slot.
# Extend this list -- or pass --prompts_file -- to scale the dataset.
DEFAULT_INTENTS: list[tuple[str, list[str]]] = [
    ("A poster of a vintage red bicycle leaning against a brick wall at sunset.", ["bicycle", "wall", "sunset"]),
    ("A serene mountain lake at dawn with mist over the water and pine trees.", ["mountain", "lake", "mist", "pine"]),
    ("A futuristic city skyline at night with neon lights and flying cars.", ["city", "night", "neon", "flying car"]),
    ("A cozy coffee shop interior in warm morning light, plants on the shelves.", ["coffee", "warm", "plants"]),
    (
        "A watercolor illustration of a fox curled up asleep in autumn leaves.",
        ["fox", "autumn", "leaves", "watercolor"],
    ),
    ("A dramatic ocean wave crashing against dark rocks under a stormy sky.", ["wave", "rocks", "storm"]),
    ("A minimalist product shot of a ceramic teapot on a white background.", ["teapot", "minimalist", "white"]),
    (
        "A whimsical hot-air balloon festival over rolling green hills at golden hour.",
        ["balloon", "hills", "golden hour"],
    ),
    ("A close-up portrait of an astronaut reflected in a space-helmet visor.", ["astronaut", "helmet", "reflection"]),
    ("A lush tropical waterfall in a jungle with sunbeams through the canopy.", ["waterfall", "jungle", "sunbeam"]),
    ("A snowy alpine village at twilight with warm lights in the windows.", ["village", "snow", "twilight"]),
    ("A vibrant street market in Marrakech with spices, textiles and lanterns.", ["market", "spices", "lantern"]),
]


def _keywords_from_prompt(prompt: str, k: int = 4) -> list[str]:
    """Derive naive keyword ground truth from a free-form prompt (content words)."""
    stop = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "over", "under", "to", "for"}
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", prompt) if w.lower() not in stop and len(w) > 3]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return seen[:k]


def _build_row(intent: str, keywords: list[str], split: str, index: int) -> pd.Series:
    ground_truth = {"keywords": keywords}
    return pd.Series(
        {
            "data_source": "image_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": intent},
            ],
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": {
                "index": index,
                "split": split,
                "intent": intent,
                "tools_kwargs": {"reward": {"ground_truth": ground_truth}},
            },
            "agent_name": "image_agent",
        }
    )


def _load_intents(prompts_file: str | None) -> list[tuple[str, list[str]]]:
    if not prompts_file:
        return DEFAULT_INTENTS
    path = os.path.expanduser(prompts_file)
    intents: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                intents.append((line, _keywords_from_prompt(line)))
    if not intents:
        raise ValueError(f"No prompts found in {path}")
    return intents


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image-generation intents for uni-agent training.")
    parser.add_argument("--prompts_file", default=None, help="Optional newline-separated intents (one per line).")
    parser.add_argument("--local_save_dir", default="./image_agent_data", help="Output dir for train/test parquet.")
    parser.add_argument("--test_rows", type=int, default=2, help="How many intents to hold out for the test split.")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat the intent list N times (to enlarge the train set)."
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    intents = _load_intents(args.prompts_file) * max(1, args.repeat)
    test_n = min(args.test_rows, max(1, len(intents) // 5))
    train_intents, test_intents = intents[test_n:], intents[:test_n]

    train_df = pd.DataFrame([_build_row(t, kw, "train", i) for i, (t, kw) in enumerate(train_intents)])
    test_df = pd.DataFrame([_build_row(t, kw, "test", i) for i, (t, kw) in enumerate(test_intents)])

    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Saved {len(train_df)} train rows to {train_path}")
    logger.info(f"Saved {len(test_df)} test rows to {test_path}")


if __name__ == "__main__":
    main()
