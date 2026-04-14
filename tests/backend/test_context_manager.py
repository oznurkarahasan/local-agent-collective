import json

from backend.core.context_manager import ContextManager


def test_sliding_window_summarizes_and_trims_buffer(tmp_path):
    manager = ContextManager(
        data_root=tmp_path / "context",
        window_size=4,
        summarize_count=2,
    )

    for idx in range(4):
        manager.append_message("s1", "user", f"message-{idx}")

    context = manager.get_context_for_prompt("s1")
    assert "Global summaries:" in context
    assert "message-0" in context
    assert "message-1" in context
    assert "Recent global messages:" in context
    assert "message-2" in context
    assert "message-3" in context

    summary_file = tmp_path / "context" / "summaries" / "s1.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert len(payload["global"]) == 1


def test_agent_scope_is_isolated_from_global_context_view(tmp_path):
    manager = ContextManager(
        data_root=tmp_path / "context",
        window_size=10,
        summarize_count=5,
    )

    manager.append_message("s2", "user", "global-msg")
    manager.append_message("s2", "assistant", "agent-msg", agent_id="rag_agent")

    global_only = manager.get_context_for_prompt("s2")
    assert "Recent global messages:" in global_only
    assert "agent-msg" in global_only
    assert "Agent summaries (rag_agent):" not in global_only

    scoped = manager.get_context_for_prompt("s2", agent_id="rag_agent")
    assert "Recent agent messages (rag_agent):" in scoped
    assert "agent-msg" in scoped


def test_raw_messages_persist_to_jsonl(tmp_path):
    manager = ContextManager(data_root=tmp_path / "context")
    manager.append_message("s3", "user", "hello")
    manager.append_message("s3", "assistant", {"status": "ok"}, agent_id="qa_agent")

    session_file = tmp_path / "context" / "sessions" / "s3.jsonl"
    lines = session_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["content"] == "hello"
    assert first["agent_id"] is None
    assert second["agent_id"] == "qa_agent"
    assert second["content"] == '{"status": "ok"}'
