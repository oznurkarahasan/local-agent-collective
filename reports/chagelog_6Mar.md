feat 1: adaptive memory integration for RAG agent

- errors.json populated with 5 known error scenarios
  (FileNotFoundError, OllamaConnectionError, ValueError, ChromaDB)
- 13 integration tests covering full memory loop:
  skill registration, success rate tracking, error logging,
  scan_errors injection, training sample generation
- preloaded errors verified on agent startup

feat 2: implement CLI interface with end-to-end integration

- welcome screen with system status check
- load command: document loading via RAG agent
- ask command: semantic search and answer generation
- list docs, status, memory stats, help, exit commands
- ChromaDB telemetry disabled
- real integration test passed: EN and TR queries working
- 20 unit tests, all passing"