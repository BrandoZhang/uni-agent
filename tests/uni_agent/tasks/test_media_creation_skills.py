"""Tests for the media_creation skills glue (discover / manifest / install).

The creation task sources its skills from media-ai (``media_ai.agent_skills_dir()``)
rather than vendoring a copy, then uploads them into the sandbox and surfaces a
progressive-disclosure manifest. These tests pin that contract offline; they
self-skip if media-ai isn't installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uni_agent.tasks.media_creation import skills as sk

media_ai = pytest.importorskip("media_ai", reason="media-ai not installed")

EXPECTED = {
    "media-ai-shared", "media-ai-image", "media-ai-video", "media-ai-speech", "media-ai-music",
    "media-ai-sound", "media-ai-job", "media-ai-concat", "media-ai-capabilities", "media-ai-usage",
}


class _RecordingSandbox:
    def __init__(self):
        self.uploads: list[tuple[str, str]] = []
        self.scripts: list[str] = []

    async def exec_shell(self, script, **kwargs):
        self.scripts.append(script)

    async def upload(self, local, remote):
        self.uploads.append((str(local), remote))


def test_resolver_finds_media_ai_skills():
    resolved = sk.media_ai_skills_dir()
    assert resolved is not None
    assert resolved == media_ai.agent_skills_dir()
    assert (resolved / "media-ai-shared" / "SKILL.md").is_file()


def test_discover_parses_name_and_description():
    skills = sk.discover_skills(sk.media_ai_skills_dir())
    names = {s.name for s in skills}
    assert EXPECTED <= names, f"missing: {EXPECTED - names}"
    shared = next(s for s in skills if s.name == "media-ai-shared")
    assert shared.description  # frontmatter description parsed
    assert shared.source_dir.name == "media-ai-shared"


def test_manifest_lists_in_sandbox_locations():
    skills = sk.discover_skills(sk.media_ai_skills_dir())
    manifest = sk.build_manifest(skills, "/opt/skills")
    assert "<available_skills>" in manifest and "</available_skills>" in manifest
    assert "<location>/opt/skills/media-ai-shared/SKILL.md</location>" in manifest
    # one <skill> block per discovered skill
    assert manifest.count("<skill>") == len(skills)


def test_install_uploads_the_tree_into_the_sandbox():
    src = sk.media_ai_skills_dir()
    sandbox = _RecordingSandbox()
    asyncio.run(sk.install_skills(sandbox, src, "/opt/skills"))
    assert sandbox.uploads == [(str(src), "/opt/skills")]
    assert sandbox.scripts and "mkdir -p" in sandbox.scripts[0]


def test_env_override_redirects_resolver(tmp_path, monkeypatch):
    fake = tmp_path / "skills"
    (fake / "media-ai-shared").mkdir(parents=True)
    (fake / "media-ai-shared" / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
    monkeypatch.setenv("CREATION_SKILLS_DIR", str(fake))
    assert sk.media_ai_skills_dir() == Path(fake)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
