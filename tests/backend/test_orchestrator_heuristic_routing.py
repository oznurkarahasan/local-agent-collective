import json
from unittest.mock import AsyncMock, MagicMock

from backend.core.orchestrator import Orchestrator


def _write_agent(base, agent_id: str) -> None:
    agent_dir = base / agent_id
    agent_dir.mkdir()
    (agent_dir / "memory").mkdir()
    (agent_dir / "config.json").write_text(
        json.dumps(
            {
                "id": agent_id,
                "name": agent_id,
                "capabilities": [],
                "input_types": [],
                "enabled": True,
            }
        )
    )
    (agent_dir / "agent.py").write_text(
        "from backend.core.agent_base import AgentBase\n"
        "class StubAgent(AgentBase):\n"
        "    def get_capabilities(self):\n"
        "        return []\n"
        "    def get_required_model_role(self):\n"
        "        return 'orchestration'\n"
        "    async def _execute(self, task):\n"
        "        return {'success': True, 'output': 'ok'}\n"
    )


def _make_orchestrator(tmp_path) -> Orchestrator:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for agent_id in ("rag_agent", "coder_agent", "research_agent", "qa_agent"):
        _write_agent(agents_dir, agent_id)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.json").write_text(
        json.dumps(
            {
                "models": [
                    {"id": "gemma4:e4b", "roles": ["orchestration"], "keep_alive": "0"},
                    {"id": "nomic-embed-text:v1.5", "roles": ["embedding"], "keep_alive": "-1"},
                ]
            }
        )
    )

    mock_ollama = MagicMock()
    mock_ollama.chat = AsyncMock(return_value='{"steps": []}')
    mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    return Orchestrator(
        agents_dir=agents_dir,
        config_dir=config_dir,
        ollama_client=mock_ollama,
    )


def test_heuristic_research_routes_to_research_not_rag(tmp_path):
    orchestrator = _make_orchestrator(tmp_path)
    plan = orchestrator._heuristic_plan("2025'te en populer yapay zeka trendi nedir?")
    assert plan is not None
    assert plan["steps"][0]["agent"] == "research_agent"
    assert all(step["agent"] != "rag_agent" for step in plan["steps"])


def test_heuristic_document_query_routes_to_rag(tmp_path):
    orchestrator = _make_orchestrator(tmp_path)
    plan = orchestrator._heuristic_plan("Yuklenen PDF belgesine gore ozet cikar.")
    assert plan is not None
    assert plan["steps"][0]["agent"] == "rag_agent"


def test_heuristic_code_query_routes_to_coder(tmp_path):
    orchestrator = _make_orchestrator(tmp_path)
    plan = orchestrator._heuristic_plan("Bu Python kodundaki bug'i bul.")
    assert plan is not None
    assert plan["steps"][0]["agent"] == "coder_agent"


def test_initialize_bootstraps_memory_files_for_all_agents(tmp_path):
    orchestrator = _make_orchestrator(tmp_path)
    # initialize() is async only because of model warmup
    import asyncio

    asyncio.run(orchestrator.initialize())

    for agent_id in ("rag_agent", "coder_agent", "research_agent", "qa_agent"):
        memory_dir = tmp_path / "agents" / agent_id / "memory"
        assert (memory_dir / "skills.json").exists()
        assert (memory_dir / "errors.json").exists()
        assert (memory_dir / "training_candidates.json").exists()
