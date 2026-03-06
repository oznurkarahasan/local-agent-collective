feat 1: adaptive memory integration for RAG agent

- errors.json populated with 5 known error scenarios
  (FileNotFoundError, OllamaConnectionError, ValueError, ChromaDB)
- 13 integration tests covering full memory loop:
  skill registration, success rate tracking, error logging,
  scan_errors injection, training sample generation
- preloaded errors verified on agent startup