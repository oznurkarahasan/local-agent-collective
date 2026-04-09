"""
coder_agent/agent.py

Coder Agent — analyzes and answers questions about code files.
Uses qwen2.5-coder:3b for code-specific understanding.
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.core.agent_base import AgentBase
from backend.core.ollama_client import OllamaClient
from backend.core.platform_utils import PlatformUtils


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".java",
    ".md",
    ".txt",
}

# Language map for syntax context
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
    ".md": "markdown",
    ".txt": "text",
}


class CoderAgent(AgentBase):
    """
    Coder Agent — loads code files and answers questions about them.

    Supported task types:
    - load_code: Load and embed a code file into ChromaDB
    - query: Answer a question using stored code
    """

    def __init__(
        self,
        agent_id: str = "coder_agent",
        memory_dir: Optional[Path] = None,
        ollama_client: Optional[OllamaClient] = None,
        chroma_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        if memory_dir is None:
            memory_dir = Path(__file__).parent / "memory"

        super().__init__(
            agent_id=agent_id,
            memory_dir=memory_dir,
            ollama_client=ollama_client,
        )

        # Load agent config
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        self.config = self._load_config(config_path)

        # ChromaDB setup — separate collection from RAG agent
        if chroma_dir is None:
            chroma_dir = PlatformUtils.get_vector_store_dir()
        self.chroma_dir = chroma_dir

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="coder_agent_docs",
            metadata={"hnsw:space": "cosine"},
        )

        # Chunking config
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.top_k = 5

        # Models
        self.embed_model = "nomic-embed-text:v1.5"
        self.chat_model = "qwen2.5-coder:3b"

    def _load_config(self, config_path: Path) -> dict:
        """Load agent configuration from config.json."""
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_capabilities(self) -> list[str]:
        return self.config.get(
            "capabilities",
            ["code_analysis", "code_explanation"],
        )

    def get_required_model_role(self) -> str:
        return "code_analysis"

    async def _execute(self, task: dict) -> dict:
        """
        Route task to appropriate handler.

        Supported types:
        - load_code: {"type": "load_code", "input": "/path/to/file.py"}
        - anything else defaults to: query {"type": "...", "input": "your question"}
        """
        task_type = task.get("type")
        task_input = task.get("input", "")

        if task_type == "load_code":
            return await self._handle_load_code(task_input)
        else:
            # The LLM planner might generate task types like 'code_explanation' based on capabilities.
            return await self._handle_query(task_input)

    # --- Code Loading ---

    async def _handle_load_code(self, file_path: str) -> dict:
        """Load a code file and store its embeddings in ChromaDB."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load code
        code_text = self.load_code(path)

        # Chunk code
        chunks = self.chunk_code(code_text, path.suffix)

        if not chunks:
            return {
                "success": False,
                "output": None,
                "error": "Code file produced no chunks",
            }

        # Embed and store
        stored_count = await self.embed_and_store(
            chunks, source=str(path), language=LANGUAGE_MAP.get(path.suffix, "text")
        )

        return {
            "success": True,
            "output": {
                "file": str(path),
                "language": LANGUAGE_MAP.get(path.suffix, "text"),
                "chunks_stored": stored_count,
            },
        }

    def load_code(self, path: Path) -> str:
        """
        Load a code file as plain text.

        Args:
            path: Path to the code file

        Returns:
            File content as string.

        Raises:
            ValueError: If file type is not supported.
            FileNotFoundError: If file does not exist.
        """
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return path.read_text(encoding="utf-8", errors="ignore")

    def chunk_code(self, code_text: str, suffix: str = ".py") -> list[str]:
        """
        Split code into overlapping chunks.

        Uses language-aware separators for better chunking.

        Args:
            code_text: Raw code content
            suffix: File extension for language-specific splitting

        Returns:
            List of code chunk strings.
        """
        # Language-aware separators
        if suffix in (".py",):
            separators = ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
        elif suffix in (".js", ".ts"):
            separators = ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""]
        elif suffix in (".go",):
            separators = ["\nfunc ", "\ntype ", "\n\n", "\n", " ", ""]
        else:
            separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
        )
        return splitter.split_text(code_text)

    async def embed_and_store(
        self, chunks: list[str], source: str, language: str = "text"
    ) -> int:
        """
        Embed code chunks and store in ChromaDB.

        Args:
            chunks: List of code chunk strings
            source: Source file path for metadata
            language: Programming language for metadata

        Returns:
            Number of chunks stored.
        """
        ids = [f"{source}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source,
                "chunk_index": i,
                "language": language,
            }
            for i in range(len(chunks))
        ]

        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            vector = await self.ollama.embed(model=self.embed_model, text=chunk)
            embeddings.append(vector)

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    # --- Querying ---

    async def _handle_query(self, question: str) -> dict:
        """Answer a question using stored code."""
        if not question.strip():
            return {
                "success": False,
                "output": None,
                "error": "Question cannot be empty",
            }

        # Search ChromaDB
        context_chunks = await self.query(question)

        if not context_chunks:
            return {
                "success": True,
                "output": "No relevant code found for your question.",
            }

        # Generate answer
        answer = await self.generate_answer(question, context_chunks)

        return {
            "success": True,
            "output": {
                "answer": answer,
                "sources": list({c["source"] for c in context_chunks}),
                "language": context_chunks[0].get("language", "unknown"),
                "chunks_used": len(context_chunks),
            },
        }

    async def query(self, question: str) -> list[dict]:
        """
        Embed question and retrieve top-k matching code chunks.

        Args:
            question: User question about the code

        Returns:
            List of dicts with 'text', 'source', 'language' keys.
        """
        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        question_vector = await self.ollama.embed(model=self.embed_model, text=question)

        results = self.collection.query(
            query_embeddings=[question_vector],
            n_results=min(self.top_k, collection_count),
            include=["documents", "metadatas"],
        )

        chunks = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "language": meta.get("language", "text"),
                }
            )

        return chunks

    async def generate_answer(self, question: str, context_chunks: list[dict]) -> str:
        """
        Generate an answer using retrieved code context and qwen2.5-coder:3b.

        Args:
            question: User question
            context_chunks: List of relevant code chunks

        Returns:
            Generated answer as string.
        """
        system_prompt = self.config.get(
            "system_prompt",
            "You are an expert software engineer. Answer based on the provided code.",
        )

        context_text = "\n\n---\n\n".join(
            [
                f"File: {c['source']} ({c.get('language', 'text')})\n{c['text']}"
                for c in context_chunks
            ]
        )

        user_message = (
            f"Code context:\n{context_text}\n\n" f"Question: {question}\n\n" "Answer:"
        )

        return await self.ollama.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            keep_alive="0",
        )
