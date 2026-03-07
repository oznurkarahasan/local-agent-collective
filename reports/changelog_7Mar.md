feat 1: implement coder agent with CLI integration

- agents/coder_agent/agent.py: code loading, chunking, embedding, semantic search
- agents/coder_agent/config.json: capabilities and model config
- agents/coder_agent/memory/skills.json: 5 initial skills
- agents/coder_agent/memory/errors.json: 5 known error scenarios
- interface/cli.py: routes .py/.js/.ts/... to CoderAgent, docs to RagAgent
- real integration test passed: EN and TR queries working with qwen2.5-coder:3b
- unit tests + integration tests, all passing

feat 2: implement web UI (FastAPI + frontend)

- frontend/api/main.py: FastAPI app with CORS, lifespan, static files
- frontend/api/routers/status.py: GET /status, GET /status/memory
- frontend/api/routers/documents.py: POST /documents/load, GET /documents
- frontend/api/routers/query.py: POST /query with auto/documents/code/both targets
- frontend/api/dependencies.py: singleton agent instances
- frontend/static/index.html: single page UI with file upload and chat interface
- infra/docker-compose.yml: web service added (port 8000)
- ci-frontend.yml: activated for frontend/** changes
- 14 API tests, 195 total tests passing

Closes #18