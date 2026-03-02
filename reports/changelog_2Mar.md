feat 1: implement RAG agent with ChromaDB integration

- config.json with capabilities and system prompt
- memory/skills.json with 5 initial skills
- memory/errors.json empty initial log
- load_document() for PDF, TXT, DOCX, MD files
- chunk_document() with configurable size and overlap
- embed_and_store() via nomic-embed-text to ChromaDB
- query() semantic search with top-k retrieval
- generate_answer() via qwen3:4b with context
- _execute() handles load_document and query task types
- auto-discovered by AgentRegistry
- 19 unit tests, all passing