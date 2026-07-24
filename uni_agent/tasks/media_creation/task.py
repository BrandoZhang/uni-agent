"""Media-creation task: turn a brief into media with the ``media-ai`` CLI.

The agent (any tool-using agent, e.g. ``react`` with the ``shell`` tool) drives
the standalone **media-ai** CLI, guided by media-ai's own packaged Agent Skills
(uploaded into the sandbox and surfaced as a progressive-disclosure manifest in
the system prompt). It is scored by a cost-aware reward: did it produce a real
film, for how many generation tokens (see :mod:`.reward`).

There is no per-command uni-agent tool and no bespoke media agent -- media-ai is
a self-driving CLI, so the generic ``shell`` tool + its skills are enough.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from . import skills as skills_mod

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a multimodal media-creation agent. You fulfil the user's creative request with the
`media-ai` command-line tool, run through the `shell` tool. There is NO fixed pipeline: read the request, decide
which commands to run and in what order, and adapt to what the user actually asked for (a single image, a one-shot
clip, a multi-shot film, an edit of a provided asset, speech/music/sound, etc.).

# Workspace
Do all work under: {workspace}
Write every asset there (pass it as `--output`) and reuse the exact paths the CLI reports back in its JSON.

# The media-ai CLI
Generation is the unified `media-ai <group> <op>` CLI (on PATH): `image generate|edit`, `video generate`,
`speech generate|dialogue`, `music generate|plan`, `sound generate`, `concat`, `job query|cancel`, `capabilities`,
`usage`. Every command prints ONE JSON object on stdout (`artifacts[]`, `usage`, `meta`) and encodes failures in
its exit code. Run `media-ai capabilities --provider <p> [--model <m>]` before committing to sizes/durations you
are unsure a model supports.

# Cost
Every generation spends tokens (see each command's `usage` block). Cost is an evaluation metric -- minimize it:
use the smallest size/resolution and shortest duration that meet the request, don't regenerate assets you already
have, and don't request extras the user didn't ask for. Run `media-ai usage` before finishing and report the total.

# Discipline
End by replying with a short plain-text summary (what you produced, where it is, and the total cost), or by
calling the `submit`/`finish` tool if you have one."""


class MediaCreationTaskConfig(TaskConfig):
    """Config for the media-creation task (extends the shared task fields)."""

    name: str = "media_creation"
    workspace: str = Field(
        default="/tmp/media_creation",
        description="In-sandbox directory the agent writes artifacts to.",
    )
    skills_root: str = Field(
        default="/tmp/media_ai_skills",
        description="In-sandbox directory media-ai's skills are uploaded to (surfaced in the prompt).",
    )
    inject_skills: bool = Field(
        default=True,
        description="Upload media-ai's skills and add a progressive-disclosure manifest to the prompt.",
    )
    cost_weight: float = Field(
        default=1e-4,
        description="Reward penalty per generation token: reward = quality - cost_weight * total_tokens.",
    )
    quality_min_seconds: int = Field(
        default=4,
        description="Footage (seconds) a film needs for full quality credit (else partial).",
    )


@register_task("media_creation")
class MediaCreationTask(Task):
    """Brief -> media, via the ``media-ai`` CLI + its skills, scored by a cost-aware reward."""

    name = "media_creation"
    config_model = MediaCreationTaskConfig

    async def run(self) -> TaskResult:
        cfg: MediaCreationTaskConfig = self.config  # type: ignore[assignment]
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(f"starting media_creation task\ntask config: {json.dumps(task_config_dump, indent=2)}")

        async with self.build_sandbox() as sandbox:
            await sandbox.exec_shell(f"mkdir -p {cfg.workspace}")
            messages = await self._build_messages(sandbox, cfg)

            agent = self.build_agent()
            # The endpoint the agent calls lives on cfg.agent.model (the agent validates it).
            agent_result = await agent.run(sandbox=sandbox, messages=messages)

            from .reward import compute_reward

            result = await compute_reward(
                cfg.metadata,
                sandbox,
                agent_result,
                cost_weight=cfg.cost_weight,
                min_seconds=cfg.quality_min_seconds,
            )
            logger.info(f"task done: reward={result['reward']:.4f} film={result['final_film']}")
            return TaskResult(reward=float(result["reward"]), accuracy=float(result["quality_proxy"]), info=result)

    async def _build_messages(self, sandbox, cfg: MediaCreationTaskConfig) -> list[dict[str, Any]]:
        """Compose the prompt: system instructions (+ skills manifest) then the brief.

        The brief is ``cfg.prompt`` (typically one user message). Any system
        message already in ``cfg.prompt`` is preserved after ours.
        """
        system = SYSTEM_PROMPT.format(workspace=cfg.workspace)

        if cfg.inject_skills:
            manifest = await self._install_and_manifest(sandbox, cfg)
            if manifest:
                system = f"{system}\n\n# Skills\n{manifest}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(cfg.prompt)
        return messages

    async def _install_and_manifest(self, sandbox, cfg: MediaCreationTaskConfig) -> str | None:
        """Upload media-ai's skills into the sandbox and return their manifest (or None)."""
        host_dir: Path | None = skills_mod.media_ai_skills_dir()
        if host_dir is None:
            logger.warning("media-ai skills not found (is media-ai installed?); running without a skills manifest")
            return None
        skills = skills_mod.discover_skills(host_dir)
        if not skills:
            return None
        await skills_mod.install_skills(sandbox, host_dir, cfg.skills_root)
        logger.info(f"installed {len(skills)} media-ai skill(s) at {cfg.skills_root}: {[s.name for s in skills]}")
        return skills_mod.build_manifest(skills, cfg.skills_root)
