"""Expose media-ai's packaged Agent Skills to a white-box agent.

media-ai ships one ``SKILL.md`` per CLI functionality and locates them via
``media_ai.agent_skills_dir()``. We upload that tree into the sandbox and build a
**progressive-disclosure manifest** for the system prompt: the agent sees each
skill's name + description + in-sandbox path, then ``cat``s the SKILL.md it needs
before running the matching ``media-ai`` command. This keeps media-ai the single
source of truth for how to drive the CLI -- uni-agent never vendors a copy.

The ``media_ai`` import is optional and lazy: environments without the package
(or its skills) resolve to ``None`` and simply run without the manifest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Env overrides checked before importing media-ai (an explicit local skills dir
# wins, e.g. for a custom/edited copy). ``CREATION_SKILLS_DIR`` is the creation
# knob; ``MEDIA_AI_SKILLS_DIR`` is media-ai's own override.
_ENV_OVERRIDES = ("CREATION_SKILLS_DIR", "MEDIA_AI_SKILLS_DIR")


@dataclass(frozen=True)
class Skill:
    """One discovered skill: its identity + the host dir holding its SKILL.md."""

    name: str
    description: str
    source_dir: Path


def media_ai_skills_dir() -> Path | None:
    """Return media-ai's packaged skills directory, or ``None`` if unavailable.

    Resolution: an env override (``CREATION_SKILLS_DIR`` / ``MEDIA_AI_SKILLS_DIR``)
    pointing at a real directory, else ``media_ai.agent_skills_dir()``.
    """
    for var in _ENV_OVERRIDES:
        raw = os.environ.get(var)
        if raw:
            path = Path(raw).expanduser()
            if path.is_dir():
                return path

    try:
        import media_ai
    except Exception:
        return None
    try:
        return media_ai.agent_skills_dir()
    except Exception:
        return None


def discover_skills(skills_dir: Path) -> list[Skill]:
    """Parse every ``<skills_dir>/<name>/SKILL.md`` into a :class:`Skill` (sorted by name)."""
    skills: list[Skill] = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        name, description = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        skills.append(
            Skill(
                name=name or skill_md.parent.name,
                description=description,
                source_dir=skill_md.parent,
            )
        )
    return skills


def build_manifest(skills: list[Skill], remote_root: str) -> str:
    """Render the ``<available_skills>`` block, with each SKILL.md's *in-sandbox* path.

    ``remote_root`` is where the skills were uploaded in the sandbox; the agent
    reads ``<location>`` with the shell tool (``cat <location>``).
    """
    root = remote_root.rstrip("/")
    lines = [
        "The following skills document the `media-ai` CLI, one per functionality.",
        "Read a skill's SKILL.md (e.g. `cat <location>`) before using its commands; "
        "start with `media-ai-shared`.",
        "",
        "<available_skills>",
    ]
    for skill in skills:
        location = f"{root}/{skill.name}/SKILL.md"
        lines.append("  <skill>")
        lines.append(f"    <name>{_xml_escape(skill.name)}</name>")
        lines.append(f"    <description>{_xml_escape(skill.description.strip() or '(no description)')}</description>")
        lines.append(f"    <location>{_xml_escape(location)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


async def install_skills(sandbox, skills_dir: Path, remote_root: str) -> None:
    """Upload the skills tree into the sandbox at ``remote_root``."""
    await sandbox.exec_shell(f"rm -rf {_sh_quote(remote_root)} && mkdir -p {_sh_quote(remote_root)}")
    await sandbox.upload(skills_dir, remote_root)


def _parse_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(name, description)`` from a SKILL.md's YAML frontmatter (both may be empty)."""
    import yaml

    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return "", ""
    end = stripped.find("\n---", 3)
    if end == -1:
        return "", ""
    try:
        meta = yaml.safe_load(stripped[3:end])
    except yaml.YAMLError:
        meta = None
    if not isinstance(meta, dict):
        return "", ""
    return str(meta.get("name") or "").strip(), str(meta.get("description") or "").strip()


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _sh_quote(path: str) -> str:
    import shlex

    return shlex.quote(path)


__all__ = ["Skill", "media_ai_skills_dir", "discover_skills", "build_manifest", "install_skills"]
