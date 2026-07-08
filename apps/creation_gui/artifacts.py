"""Artifact registry: map a generated file to an opaque, servable URL.

Keeps the GUI from ever seeing filesystem paths and — critically — confines
what can be served to a single root directory (the GUI's sessions dir), so a
tool observation can't trick the server into serving arbitrary host files.
Pure stdlib; unit-testable without the loop.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._tokens: dict[str, Path] = {}

    def register(self, path: str | Path) -> str | None:
        """Register a path and return ``/api/artifacts/<token>``.

        Returns ``None`` if the resolved path escapes ``root`` (symlink or
        ``..`` traversal) — such a path is never served.
        """
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            return None
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None  # outside the served root -> refuse
        token = hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
        self._tokens[token] = resolved
        return f"/api/artifacts/{token}"

    def resolve(self, token: str) -> Path | None:
        """Resolve a previously-registered token to its path (or ``None``).

        Only tokens minted by :meth:`register` resolve, and each was already
        confined to ``root``, so this can't be used to walk the filesystem.
        """
        path = self._tokens.get(token)
        if path is None:
            return None
        # Re-verify confinement at serve time (defence in depth).
        try:
            path.resolve().relative_to(self.root)
        except (OSError, ValueError):
            return None
        return path
