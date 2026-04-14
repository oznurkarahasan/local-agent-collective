import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.rag_agent.agent import RagAgent
from backend.core.model_registry import ModelRegistry
from backend.core.ollama_client import OllamaClient

async def test_model_resolution():
    print("Testing RagAgent model resolution...")
    
    config_dir = Path(__file__).parent.parent / "config"
    registry = ModelRegistry(config_path=config_dir / "models.json")
    ollama = OllamaClient()
    
    agent = RagAgent(
        agent_id="rag_agent",
        memory_dir=Path(__file__).parent / "test_memory",
        ollama_client=ollama,
        model_registry=registry
    )
    
    print(f"Agent ID: {agent.agent_id}")
    print(f"Embed Model: {agent.embed_model}")
    print(f"Chat Model: {agent.chat_model}")
    
    expected_chat = "llama3.2:3b"
    if agent.chat_model == expected_chat:
        print("SUCCESS: Chat model correctly resolved to llama3.2:3b")
    else:
        print(f"FAILURE: Chat model is {agent.chat_model}, expected {expected_chat}")

if __name__ == "__main__":
    asyncio.run(test_model_resolution())
