"""Creation-agent GUI server (stdlib only).

A thin adapter that runs the real ``AgentInteraction`` loop server-side and
streams *framework-neutral* events to a browser over SSE. The browser
(``static/index.html``) depends only on this HTTP contract — never on
uni-agent — so the same GUI works over any backend that speaks these events.

Endpoints:
    GET  /                              -> the GUI
    POST /api/sessions                  {brief, backend?}   -> {session_id}
    GET  /api/sessions/<id>/stream      -> SSE event stream (see events.py)
    POST /api/sessions/<id>/cancel      -> cooperative cancel
    GET  /api/artifacts/<token>         -> generated image/video bytes (Range-aware)

Run:  python -m apps.creation_gui.server --port 8770
Offline by default (mock LLM + mock media backend); set BASE_URL / MEDIA_BACKEND=volc
(+ ARK_API_KEY) for real generation.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.creation_gui.artifacts import ArtifactRegistry  # noqa: E402
from apps.creation_gui.events import translate_step  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
_STREAM_SENTINEL = object()  # queued to signal "run finished, close the stream"


@dataclass
class Session:
    id: str
    workspace: Path
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class SessionManager:
    def __init__(self, sessions_root: Path, backend: str, base_url: str | None):
        self.sessions_root = sessions_root
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.base_url = base_url
        self.registry = ArtifactRegistry(self.sessions_root)
        self._sessions: dict[str, Session] = {}

    def create(self, brief: str, backend: str | None = None) -> Session:
        sid = uuid.uuid4().hex[:12]
        session = Session(id=sid, workspace=self.sessions_root / sid)
        self._sessions[sid] = session

        def on_step(step_output, metrics):
            for event in translate_step(step_output, metrics, self.registry.register):
                session.events.put(event)

        def worker():
            from apps.creation_gui.session import run_session

            try:
                run_session(
                    brief=brief,
                    workspace=session.workspace,
                    backend=(backend or self.backend),
                    base_url=self.base_url,
                    on_step=on_step,
                    should_cancel=session.cancelled.is_set,
                )
            except Exception as exc:  # noqa: BLE001 - surface failures to the UI
                session.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                session.events.put(_STREAM_SENTINEL)

        session.thread = threading.Thread(target=worker, daemon=True)
        session.thread.start()
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)


class _Handler(BaseHTTPRequestHandler):
    manager: SessionManager

    def log_message(self, *args):  # quiet
        pass

    # ---- helpers ----
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # ---- routing ----
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._serve_static("index.html")
        if path.startswith("/api/artifacts/"):
            return self._serve_artifact(path.rsplit("/", 1)[-1])
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "sessions" and parts[3] == "stream":
            return self._serve_stream(parts[2])
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/sessions":
            body = self._read_json()
            brief = (body.get("brief") or "").strip()
            if not brief:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "brief is required"})
            session = self.manager.create(brief, backend=body.get("backend"))
            return self._send_json(HTTPStatus.OK, {"session_id": session.id})
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "sessions" and parts[3] == "cancel":
            session = self.manager.get(parts[2])
            if session is None:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
            session.cancelled.set()
            return self._send_json(HTTPStatus.OK, {"ok": True, "cancelled": True})
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    # ---- handlers ----
    def _serve_static(self, name: str) -> None:
        file = STATIC_DIR / name
        if not file.is_file():
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "missing static asset"})
        data = file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_stream(self, sid: str) -> None:
        session = self.manager.get(sid)
        if session is None:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    item = session.events.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")  # comment ping
                    self.wfile.flush()
                    continue
                if item is _STREAM_SENTINEL:
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client closed the tab

    def _serve_artifact(self, token: str) -> None:
        path = self.manager.registry.resolve(token)
        if path is None or not path.is_file():
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown artifact"})
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes=") :].split("-", 1)
            try:
                if spec[0]:
                    start = int(spec[0])
                if len(spec) > 1 and spec[1]:
                    end = int(spec[1])
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            start = max(0, min(start, end))
        partial = bool(rng) and (start, end) != (0, size - 1)
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


def build_server(host: str, port: int, sessions_root: Path, backend: str, base_url: str | None) -> ThreadingHTTPServer:
    manager = SessionManager(sessions_root=sessions_root, backend=backend, base_url=base_url)
    handler = type("BoundHandler", (_Handler,), {"manager": manager})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    ap = argparse.ArgumentParser(description="Creation-agent GUI server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--sessions-root", default=os.getenv("CREATION_GUI_ROOT", "/tmp/creation_gui"))
    ap.add_argument("--backend", default=os.getenv("MEDIA_BACKEND", "mock"), help="mock (offline) or volc")
    ap.add_argument(
        "--base-url", default=os.getenv("BASE_URL"), help="OpenAI-compatible LLM endpoint; omit for mock LLM"
    )
    args = ap.parse_args()

    httpd = build_server(args.host, args.port, Path(args.sessions_root), args.backend, args.base_url)
    llm = args.base_url or "scripted mock LLM"
    print(f"creation GUI: http://{args.host}:{args.port}  (backend={args.backend}, llm={llm})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
