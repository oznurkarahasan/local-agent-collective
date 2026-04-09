"""
rag_agent/agent.py

RAG (Retrieval-Augmented Generation) Agent.
Loads documents, stores them in ChromaDB, and answers questions
using semantic search + local LLM.
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.core.agent_base import AgentBase
from backend.core.ollama_client import OllamaClient
from backend.core.platform_utils import PlatformUtils


class RagAgent(AgentBase):
    """
    RAG Agent — loads documents and answers questions.

    Supported task types:
    - load_document: Load and embed a document into ChromaDB
    - query: Answer a question using stored documents
    """

    def __init__(
        self,
        agent_id: str = "rag_agent",
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

        # ChromaDB setup
        if chroma_dir is None:
            chroma_dir = PlatformUtils.get_vector_store_dir()
        self.chroma_dir = chroma_dir

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_agent_docs",
            metadata={"hnsw:space": "cosine"},
        )

        # Text splitter config
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.top_k = 5

        # Embedding model
        self.embed_model = "nomic-embed-text:v1.5"

        # Chat model
        self.chat_model = "qwen3:4b"

    def _load_config(self, config_path: Path) -> dict:
        """Load agent configuration from config.json."""
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_capabilities(self) -> list[str]:
        return self.config.get(
            "capabilities",
            ["document_qa", "semantic_search"],
        )

    def get_required_model_role(self) -> str:
        return "rag"

    async def _execute(self, task: dict) -> dict:
        """
        Route task to appropriate handler based on task type.

        Supported types:
        - load_document: {"type": "load_document", "input": "/path/to/file"}
        - anything else defaults to: query {"type": "...", "input": "your question"}
        """
        task_type = task.get("type")
        task_input = task.get("input", "")

        if task_type == "load_document":
            return await self._handle_load_document(task_input)
        else:
            # LLM planner generated task types like 'document_qa' fall through to general query
            return await self._handle_query(task_input)

    # --- Document Loading ---

    async def _handle_load_document(self, file_path: str) -> dict:
        """Load a document and store its embeddings in ChromaDB."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load document
        docs = self.load_document(path)

        # Chunk document
        chunks = self.chunk_document(docs)

        if not chunks:
            return {
                "success": False,
                "output": None,
                "error": "Document produced no chunks",
            }

        # Embed and store
        stored_count = await self.embed_and_store(chunks, source=str(path))

        return {
            "success": True,
            "output": {
                "file": str(path),
                "chunks_stored": stored_count,
            },
        }

    def load_document(self, path: Path) -> list:
        """
        Load a document using the appropriate LangChain loader.

        Args:
            path: Path to the document file

        Returns:
            List of LangChain Document objects.

        Raises:
            ValueError: If file type is not supported.
        """
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".txt" or suffix == ".md":
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                "Supported: .pdf, .txt, .md, .docx"
            )

        return loader.load()

    def chunk_document(self, docs: list) -> list:
        """
        Split documents into overlapping chunks.

        Args:
            docs: List of LangChain Document objects

        Returns:
            List of chunked LangChain Document objects.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return splitter.split_documents(docs)

    async def embed_and_store(self, chunks: list, source: str) -> int:
        """
        Embed chunks and store in ChromaDB.

        Args:
            chunks: List of LangChain Document chunks
            source: Source file path for metadata

        Returns:
            Number of chunks stored.
        """
        texts = [chunk.page_content for chunk in chunks]
        ids = [f"{source}__chunk_{i}" for i in range(len(texts))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(texts))]

        # Generate embeddings
        embeddings = []
        for text in texts:
            vector = await self.ollama.embed(model=self.embed_model, text=text)
            embeddings.append(vector)

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(texts)

    # --- Querying ---

    async def _handle_query(self, question: str) -> dict:
        """Answer a question using stored documents."""
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
                "output": "No relevant documents found for your question.",
            }

        # Generate answer
        answer = await self.generate_answer(question, context_chunks)

        return {
            "success": True,
            "output": {
                "answer": answer,
                "sources": list({c["source"] for c in context_chunks}),
                "chunks_used": len(context_chunks),
            },
        }

    async def query(self, question: str) -> list[dict]:
        """
        Embed question and retrieve top-k matching chunks from ChromaDB.

        Args:
            question: User question

        Returns:
            List of dicts with 'text' and 'source' keys.
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
                }
            )

        return chunks

    async def generate_answer(self, question: str, context_chunks: list[dict]) -> str:
        """
        Generate an answer using retrieved context and qwen3:4b.

        Args:
            question: User question
            context_chunks: List of relevant text chunks

        Returns:
            Generated answer as string.
        """
        system_prompt = self.config.get(
            "system_prompt",
            "You are a helpful assistant. Answer based on the provided context.",
        )

        context_text = "\n\n---\n\n".join(
            [f"Source: {c['source']}\n{c['text']}" for c in context_chunks]
        )

        user_message = (
            f"Context:\n{context_text}\n\n" f"Question: {question}\n\n" "Answer:"
        )

        return await self.ollama.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            keep_alive="0",
        )
