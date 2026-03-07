"""
Command-line interface for the Local AI Agent Collective.
Provides document loading, querying, and system status commands.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path before local imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from backend.core.platform_utils import PlatformUtils  # noqa: E402
from backend.core.ollama_client import OllamaClient  # noqa: E402
from agents.rag_agent.agent import RagAgent  # noqa: E402
from agents.coder_agent.agent import CoderAgent  # noqa: E402


WELCOME = """
╔══════════════════════════════════════════════════════╗
║           Indis.ai — Independent Intelligence        ║
║              Local AI Agent Collective               ║
╚══════════════════════════════════════════════════════╝

Commands:
  load <file>     Load a document or code file
  ask <question>  Ask a question about loaded files
  list docs       Show loaded files
  status          Show system status
  memory stats    Show agent memory statistics
  help            Show this help message
  exit            Exit the program
"""

HELP = """
Commands:
  load <file>     Load a document (PDF, TXT, DOCX, MD) or code file (PY, JS, TS, GO, ...)
  ask <question>  Ask a question about loaded files
  list docs       Show loaded files
  status          Show system status
  memory stats    Show agent memory statistics
  help            Show this help message
  exit            Exit the program
"""

# Extensions handled by CoderAgent
CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".java"}

# Extensions handled by RagAgent
DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class CLI:
    """Interactive CLI for the Local AI Agent Collective."""

    def __init__(self):
        self.ollama = OllamaClient()
        self.rag_agent = None
        self.coder_agent = None
        self.loaded_docs: list[str] = []
        self.loaded_code: list[str] = []

    async def start(self):
        """Start the CLI."""
        print(WELCOME)
        await self._check_system()
        await self._init_agents()
        await self._loop()

    async def _check_system(self):
        """Check system status on startup."""
        print("Checking system...")

        ollama_ok = await self.ollama.ping()
        if ollama_ok:
            print("  ✓ Ollama is running")
        else:
            print("  ✗ Ollama is not running")
            print(f"  → {PlatformUtils.get_ollama_install_instructions()}")
            print("\nPlease start Ollama and try again.")
            sys.exit(1)

        models = await self.ollama.list_models()
        model_ids = {m.get("name", "") for m in models}

        required = ["nomic-embed-text:v1.5", "qwen3:4b", "qwen2.5-coder:3b"]
        for model in required:
            if model in model_ids:
                print(f"  ✓ {model}")
            else:
                print(f"  ✗ {model} not found")
                print(f"  → Run: ollama pull {model}")

        print()

    async def _init_agents(self):
        """Initialize agents."""
        chroma_dir = PlatformUtils.get_vector_store_dir()

        rag_memory = Path(__file__).parent.parent / "agents" / "rag_agent" / "memory"
        self.rag_agent = RagAgent(
            memory_dir=rag_memory,
            ollama_client=self.ollama,
            chroma_dir=chroma_dir,
        )
        print("  ✓ RAG Agent ready")

        coder_memory = (
            Path(__file__).parent.parent / "agents" / "coder_agent" / "memory"
        )
        self.coder_agent = CoderAgent(
            memory_dir=coder_memory,
            ollama_client=self.ollama,
            chroma_dir=chroma_dir,
        )
        print("  ✓ Coder Agent ready")
        print()

    async def _loop(self):
        """Main command loop."""
        while True:
            try:
                user_input = input("indis> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            await self._handle_command(user_input)

    async def _handle_command(self, user_input: str):
        """Route command to appropriate handler."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        elif command == "help":
            print(HELP)

        elif command == "load":
            await self._cmd_load(args)

        elif command == "ask":
            await self._cmd_ask(args)

        elif command == "list":
            if args.strip() == "docs":
                self._cmd_list_docs()
            else:
                print("Unknown command. Did you mean 'list docs'?")

        elif command == "status":
            await self._cmd_status()

        elif command == "memory":
            if args.strip() == "stats":
                self._cmd_memory_stats()
            else:
                print("Unknown command. Did you mean 'memory stats'?")

        else:
            print(f"Unknown command: '{command}'. Type 'help' for commands.")

    async def _cmd_load(self, file_path: str):
        """Load a document or code file into the appropriate agent."""
        if not file_path:
            print("Usage: load <file_path>")
            return

        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return

        suffix = path.suffix.lower()

        if suffix in CODE_EXTENSIONS:
            await self._load_code(path)
        elif suffix in DOC_EXTENSIONS:
            await self._load_document(path)
        else:
            print(
                f"  ✗ Unsupported file type: '{suffix}'"
                f"\n  Documents: {', '.join(sorted(DOC_EXTENSIONS))}"
                f"\n  Code:      {', '.join(sorted(CODE_EXTENSIONS))}"
            )

    async def _load_document(self, path: Path):
        """Load a document via RAG Agent."""
        print(f"Loading {path.name} (document)...")
        result = await self.rag_agent.run(
            {"type": "load_document", "input": str(path.absolute())}
        )
        if result["success"]:
            chunks = result["output"]["chunks_stored"]
            print(f"  ✓ Loaded {path.name} ({chunks} chunks stored)")
            if str(path.absolute()) not in self.loaded_docs:
                self.loaded_docs.append(str(path.absolute()))
        else:
            print(f"  ✗ Failed to load: {result.get('error')}")

    async def _load_code(self, path: Path):
        """Load a code file via Coder Agent."""
        print(f"Loading {path.name} (code)...")
        result = await self.coder_agent.run(
            {"type": "load_code", "input": str(path.absolute())}
        )
        if result["success"]:
            chunks = result["output"]["chunks_stored"]
            lang = result["output"].get("language", "")
            print(f"  ✓ Loaded {path.name} ({lang}, {chunks} chunks stored)")
            if str(path.absolute()) not in self.loaded_code:
                self.loaded_code.append(str(path.absolute()))
        else:
            print(f"  ✗ Failed to load: {result.get('error')}")

    async def _cmd_ask(self, question: str):
        """Ask a question — routes to the agent with loaded files."""
        if not question:
            print("Usage: ask <question>")
            return

        has_docs = bool(self.loaded_docs)
        has_code = bool(self.loaded_code)

        if not has_docs and not has_code:
            print("No files loaded. Use 'load <file>' first.")
            return

        print("Thinking...")

        # If both agents have files, query both and combine
        if has_docs and has_code:
            await self._ask_both(question)
        elif has_code:
            await self._ask_agent(self.coder_agent, question, "query")
        else:
            await self._ask_agent(self.rag_agent, question, "query")

    async def _ask_agent(self, agent, question: str, task_type: str):
        """Query a single agent and print the result."""
        result = await agent.run({"type": task_type, "input": question})

        if result["success"]:
            output = result["output"]
            if isinstance(output, dict):
                print(f"\n{output['answer']}\n")
                sources = output.get("sources", [])
                if sources:
                    print(f"Sources: {', '.join(Path(s).name for s in sources)}")
                print(f"Chunks used: {output.get('chunks_used', 0)}\n")
            else:
                print(f"\n{output}\n")
        else:
            print(f"  ✗ Query failed: {result.get('error')}")

    async def _ask_both(self, question: str):
        """Query both agents and print combined results."""
        rag_result = await self.rag_agent.run({"type": "query", "input": question})
        code_result = await self.coder_agent.run({"type": "query", "input": question})

        if rag_result["success"] and isinstance(rag_result["output"], dict):
            print(f"\n[Documents]\n{rag_result['output']['answer']}\n")

        if code_result["success"] and isinstance(code_result["output"], dict):
            print(f"\n[Code]\n{code_result['output']['answer']}\n")

    def _cmd_list_docs(self):
        """List all loaded files."""
        if not self.loaded_docs and not self.loaded_code:
            print("No files loaded.")
            return

        if self.loaded_docs:
            print(f"\nDocuments ({len(self.loaded_docs)}):")
            for doc in self.loaded_docs:
                print(f"  • {Path(doc).name} ({doc})")

        if self.loaded_code:
            print(f"\nCode files ({len(self.loaded_code)}):")
            for code in self.loaded_code:
                print(f"  • {Path(code).name} ({code})")
        print()

    async def _cmd_status(self):
        """Show system status."""
        print("\nSystem Status:")

        ollama_ok = await self.ollama.ping()
        print(f"  Ollama:       {'✓ running' if ollama_ok else '✗ not running'}")
        print(f"  Platform:     {PlatformUtils.OS}")
        print(f"  Data dir:     {PlatformUtils.get_base_dir()}")
        print(f"  Vector DB:    {PlatformUtils.get_vector_store_dir()}")

        if ollama_ok:
            models = await self.ollama.list_models()
            print(f"  Models:       {len(models)} installed")

        rag_info = self.rag_agent.get_agent_info()
        print(f"  RAG Agent:    ✓ ready ({len(rag_info['skills'])} skills)")

        coder_info = self.coder_agent.get_agent_info()
        print(f"  Coder Agent:  ✓ ready ({len(coder_info['skills'])} skills)")
        print()

    def _cmd_memory_stats(self):
        """Show memory stats for all agents."""
        for label, agent in [
            ("RAG Agent", self.rag_agent),
            ("Coder Agent", self.coder_agent),
        ]:
            print(f"\n{label} Memory Stats:")
            skills = agent.memory.get_skills()
            errors = agent.memory.get_errors()

            print(f"\n  Skills ({len(skills)}):")
            for skill in skills:
                rate = skill.get("success_rate", 0)
                runs = skill.get("total_runs", 0)
                bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
                print(f"    {skill['name']:<20} [{bar}] {rate:.0%} ({runs} runs)")

            print(f"\n  Errors ({len(errors)}):")
            resolved = sum(1 for e in errors if e.get("resolved"))
            print(f"    Total: {len(errors)}, Resolved: {resolved}")
        print()


def main():
    """Entry point."""
    cli = CLI()
    asyncio.run(cli.start())


if __name__ == "__main__":
    main()
