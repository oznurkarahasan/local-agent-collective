"""
Context manager for orchestrator-level conversation memory.

Stores:
- Raw session messages in JSONL files
- Compressed summaries in JSON files
- In-memory sliding windows for active context
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class _SessionState:
    global_buffer: list[dict]
    agent_buffers: dict[str, list[dict]]
    global_summaries: list[str]
    agent_summaries: dict[str, list[str]]


class ContextManager:
    """Manages global and per-agent context with sliding windows and summaries."""

    def __init__(
        self,
        data_root: Path | None = None,
        window_size: int = 20,
        summarize_count: int = 10,
    ) -> None:
        if summarize_count <= 0:
            raise ValueError("summarize_count must be > 0")
        if window_size <= summarize_count:
            raise ValueError("window_size must be greater than summarize_count")

        requested_root = data_root or (
            Path(__file__).parent.parent.parent / "data" / "context"
        )
        base_root = self._resolve_writable_root(requested_root)
        self.sessions_dir = base_root / "sessions"
        self.summaries_dir = base_root / "summaries"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.summarize_count = summarize_count
        self._state: dict[str, _SessionState] = {}
        self._lock = Lock()

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_id: str | None = None,
    ) -> None:
        """Append a message to global scope and optionally to an agent scope."""
        normalized = (
            content
            if isinstance(content, str)
            else json.dumps(content, ensure_ascii=False)
        )
        if not normalized:
            return

        with self._lock:
            state = self._load_or_create_state(session_id)
            message = {"role": role, "content": normalized}

            state.global_buffer.append(message)
            self._maybe_summarize_scope(state.global_buffer, state.global_summaries)

            if agent_id:
                buffer_ref = state.agent_buffers.setdefault(agent_id, [])
                summaries_ref = state.agent_summaries.setdefault(agent_id, [])
                buffer_ref.append(message)
                self._maybe_summarize_scope(buffer_ref, summaries_ref)

            try:
                self._append_jsonl(
                    self.sessions_dir / f"{session_id}.jsonl",
                    {
                        "role": role,
                        "content": normalized,
                        "agent_id": agent_id,
                    },
                )
                self._write_summaries(session_id, state)
            except OSError as exc:
                logger.warning(
                    "Context persistence failed for session '%s': %s", session_id, exc
                )

    def get_context_for_prompt(
        self, session_id: str, agent_id: str | None = None
    ) -> str:
        """Build context string from summaries and active window messages."""
        with self._lock:
            state = self._load_or_create_state(session_id)
            return self._render_context(state, agent_id)

    def _load_or_create_state(self, session_id: str) -> _SessionState:
        state = self._state.get(session_id)
        if state is not None:
            return state

        summaries_payload = self._read_summaries(
            self.summaries_dir / f"{session_id}.json"
        )
        state = _SessionState(
            global_buffer=[],
            agent_buffers={},
            global_summaries=summaries_payload.get("global", []),
            agent_summaries=summaries_payload.get("agents", {}),
        )
        self._state[session_id] = state
        return state

    def _maybe_summarize_scope(
        self, buffer_ref: list[dict], summaries_ref: list[str]
    ) -> None:
        if len(buffer_ref) < self.window_size:
            return

        chunk = buffer_ref[: self.summarize_count]
        del buffer_ref[: self.summarize_count]
        summaries_ref.append(self._summarize_messages(chunk))

    def _summarize_messages(self, messages: list[dict]) -> str:
        # Deterministic lightweight summarization to keep runtime local-only.
        entries = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = " ".join((msg.get("content") or "").split())
            if len(content) > 180:
                content = content[:177] + "..."
            entries.append(f"{role}: {content}")
        return " | ".join(entries)

    def _render_context(self, state: _SessionState, agent_id: str | None) -> str:
        lines: list[str] = []
        if state.global_summaries:
            lines.append("Global summaries:")
            for item in state.global_summaries:
                lines.append(f"- {item}")
            lines.append("")

        if state.global_buffer:
            lines.append("Recent global messages:")
            for msg in state.global_buffer:
                lines.append(f'{msg.get("role", "unknown")}: {msg.get("content", "")}')
            lines.append("")

        if agent_id:
            agent_summaries = state.agent_summaries.get(agent_id, [])
            agent_buffer = state.agent_buffers.get(agent_id, [])

            if agent_summaries:
                lines.append(f"Agent summaries ({agent_id}):")
                for item in agent_summaries:
                    lines.append(f"- {item}")
                lines.append("")

            if agent_buffer:
                lines.append(f"Recent agent messages ({agent_id}):")
                for msg in agent_buffer:
                    lines.append(
                        f'{msg.get("role", "unknown")}: {msg.get("content", "")}'
                    )

        return "\n".join(lines).strip()

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_summaries(self, path: Path) -> dict:
        if not path.exists():
            return {"global": [], "agents": {}}
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            global_summaries = payload.get("global", [])
            agent_summaries = payload.get("agents", {})
            if not isinstance(global_summaries, list):
                global_summaries = []
            if not isinstance(agent_summaries, dict):
                agent_summaries = {}
            return {"global": global_summaries, "agents": agent_summaries}
        except (OSError, json.JSONDecodeError):
            return {"global": [], "agents": {}}

    def _write_summaries(self, session_id: str, state: _SessionState) -> None:
        payload = {
            "global": state.global_summaries,
            "agents": state.agent_summaries,
        }
        path = self.summaries_dir / f"{session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _resolve_writable_root(self, preferred_root: Path) -> Path:
        """
        Return a writable context root.

        If the preferred root is not writable, fall back to ~/.indis/context.
        """
        try:
            preferred_root.mkdir(parents=True, exist_ok=True)
            probe = preferred_root / ".write_probe"
            with probe.open("w", encoding="utf-8") as f:
                f.write("ok")
            probe.unlink(missing_ok=True)
            return preferred_root
        except OSError as exc:
            fallback = Path.home() / ".indis" / "context"
            logger.warning(
                "Context path '%s' is not writable (%s). Falling back to '%s'.",
                preferred_root,
                exc,
                fallback,
            )
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
