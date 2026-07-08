"""Tests for the artifact registry (path->URL with root confinement)."""

from __future__ import annotations

from apps.creation_gui.artifacts import ArtifactRegistry


def test_registers_and_resolves_path_under_root(tmp_path):
    reg = ArtifactRegistry(tmp_path)
    f = tmp_path / "sess" / "final.mp4"
    f.parent.mkdir()
    f.write_bytes(b"video")
    url = reg.register(f)
    assert url and url.startswith("/api/artifacts/")
    token = url.rsplit("/", 1)[-1]
    assert reg.resolve(token) == f.resolve()


def test_refuses_path_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    reg = ArtifactRegistry(root)
    assert reg.register(outside) is None  # escapes the served root
    assert reg.register("/etc/passwd") is None


def test_refuses_traversal_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    reg = ArtifactRegistry(root)
    assert reg.register(root / ".." / "etc" / "passwd") is None


def test_unknown_token_resolves_to_none(tmp_path):
    reg = ArtifactRegistry(tmp_path)
    assert reg.resolve("deadbeef") is None


def test_same_path_is_stable_token(tmp_path):
    reg = ArtifactRegistry(tmp_path)
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    assert reg.register(f) == reg.register(f)  # deterministic
