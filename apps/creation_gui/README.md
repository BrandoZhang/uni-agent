# Creation Agent GUI

A tiny web UI for the multimodal creation agent: type a brief, watch the agent
plan a storyboard and call its generation tools, see the generated images/clips
**inline** as they land, and track **token cost live** — all streamed.

It runs **fully offline** by default (scripted mock LLM + mock media backend);
flip two env vars for a real model + real generation.

> Continuing this work? See [`docs/creation-agent-handoff.md`](../../docs/creation-agent-handoff.md)
> for design decisions, the media-ai repo split, and open next-steps.

```bash
python -m apps.creation_gui.server --port 8770
# open http://127.0.0.1:8770
```

## Design: the GUI is decoupled from uni-agent

The browser talks to the server over a **small, framework-neutral HTTP + SSE
contract** — it never imports or knows about uni-agent. Only one server-side
module (`session.py`) touches uni-agent, and it does so through the **public**
`AgentInteraction.step()` (no change to the framework core). Swap uni-agent for
another agent framework by rewriting just `session.py` to emit the same events;
the GUI (`static/index.html`) and the event/artifact helpers stay identical.

```
 Browser (static/index.html)      Agent Server (this app)            uni-agent + media-ai
 ─ zero uni-agent dependency ─►  ┌───────────────────────────┐  ─►  ┌───────────────────┐
   POST /api/sessions            │ session.py  (only coupled  │      │ AgentInteraction   │
   EventSource .../stream  ◄───  │   layer; drives step())    │      │   .step()  (public)│
   <img>/<video> ◄─ /artifacts   │ events.py / artifacts.py   │      │ media-ai CLIs      │
                                  │   (pure, framework-neutral)│      └───────────────────┘
                                  └───────────────────────────┘
```

Why not just point the browser at uni-agent's OpenAI-compatible gateway? Because
in the OpenAI protocol **tool execution happens on the client**. The media tools
must run server-side (they touch the sandbox + the Volc API), so the server runs
the loop and streams *events*; the boundary is an event stream, not the model
API. That keeps the GUI a dumb renderer.

## HTTP contract

| Method + path | Purpose |
|---|---|
| `POST /api/sessions` `{brief, backend?}` | start a run → `{session_id}` |
| `GET  /api/sessions/<id>/stream` | SSE stream of UI events (below) |
| `POST /api/sessions/<id>/cancel` | cooperative cancel (checked between steps) |
| `GET  /api/artifacts/<token>` | generated image/video bytes (Range-aware) |

**Event types** (`type` discriminates; see `events.py`):
`assistant` (model text) · `tool_call` (name + command) · `tool_result`
(status + optional `artifact:{url,kind}` + `usage`) · `cost`
(`llm{prompt,cached,completion,total}`, `by_tool`, `total_tokens`) · `done` ·
`error`. Artifacts are opaque `/api/artifacts/<token>` URLs confined to the
sessions dir — the browser never sees a filesystem path.

## Real model + real generation

```bash
BASE_URL=http://localhost:8000/v1 MEDIA_BACKEND=volc ARK_API_KEY=<key> \
  python -m apps.creation_gui.server --port 8770
# or: python -m apps.creation_gui.server --backend volc --base-url http://localhost:8000/v1
```

The offline mock LLM plays a fixed storyboard regardless of the brief (enough to
exercise the whole streaming/preview/cost path); a real endpoint makes briefs
open-ended.

## Files

```
apps/creation_gui/
├── server.py           ← stdlib HTTP: sessions, SSE, artifact serving
├── session.py          ← the only uni-agent-coupled layer (drives step())
├── events.py           ← pure step→UI-event translation (framework-neutral)
├── artifacts.py        ← path→URL registry with root confinement
├── static/index.html   ← the GUI (zero uni-agent dependency)
└── README.md
```

## Not in v1 (deliberately)

Cancel is cooperative (between steps), so a real in-flight Volc video isn't
interrupted mid-render — that needs the async `video_task --op cancel` lever
wired through. Multi-viewer per session, auth, and a persisted gallery are out
of scope for the demo.
