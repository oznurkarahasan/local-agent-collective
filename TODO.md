# TODO — Local AI Agent Collective

Full task list, priority order, and progress tracking.
When a task is completed, change `[ ]` to `[x]` and add the related PR number.

---

## Phase 0 — Monorepo Infrastructure

> Goal: Set up GitHub repo structure, branch protections, Docker foundation and CI/CD pipeline.
> Branch: `chore/project-setup`

### Monorepo Folder Structure
- [x] Create base folder skeleton
- [x] Add `.gitkeep` to empty folders
- [x] Create `.gitignore`
- [x] Add `LICENSE` (MIT)
- [x] Create `README.md`
- [x] Create `TODO.md`

### Config Files
- [X] Create `config/models.json` — 4 model definitions:
  - `deepseek-r1:7b` → roles: orchestration, planning, reasoning
  - `qwen3:4b` → roles: rag, reasoning, multilingual
  - `qwen2.5-coder:3b` → roles: code_analysis, code_generation
  - `nomic-embed-text:v1.5` → roles: embedding
- [X] Create `config/system.json` — `max_concurrent: 2`, `keep_alive: "0"`

### Docker
- [X] Write `infra/Dockerfile` (Python 3.11, dependencies, app code)
- [X] Write `infra/docker-compose.yml`:
  - `app` service
  - `ollama` service
  - Volume definitions (ollama models persistent, data/ persistent)
  - Health check definitions
  - Network definition
- [X] Create `requirements.txt`
- [X] Test clean run with `docker compose up --build`

### GitHub Actions
- [X] Write `.github/workflows/ci-backend.yml`:
  - Trigger: `push/PR` + `paths: backend/**`
  - Steps: checkout → python setup → pip install → flake8 → black --check → pytest tests/backend/
- [X] Write `.github/workflows/ci-agents.yml`:
  - Trigger: `push/PR` + `paths: agents/**`
  - Steps: checkout → python setup → pip install → flake8 → pytest tests/agents/
- [X] Write `.github/workflows/docker-build.yml`:
  - Trigger: `push` to `main` or `dev`
  - Steps: Docker build check
- [X] Write `.github/workflows/ci-frontend.yml` (placeholder for future phase)

---

## Phase 1 — Core Layers

> Goal: Write and test the foundational modules the entire system is built upon.
> Each module is started only after the previous one is tested.
> Branch: `feature/core-layer`

### `backend/core/platform_utils.py`
- [x] `get_base_dir()` → platform-specific data directory (Linux/macOS/Windows)
- [x] `get_ollama_url()` → `http://localhost:11434`
- [x] `is_ollama_running()` → service check (systemctl / tasklist)
- [x] `get_ollama_install_instructions()` → platform-specific instructions
- [x] `tests/backend/test_platform_utils.py`

### `backend/core/ollama_client.py`
- [x] Async HTTP client (`httpx`)
- [x] `ping()` → is Ollama reachable?
- [x] `list_models()` → installed models
- [x] `pull_model(model_id)` → download model
- [x] `chat(model, messages, keep_alive="0")` → completion request
- [x] `embed(model, text)` → embedding request
- [x] `unload_model(model_id)` → drop model from memory
- [x] Connection error handling (retry + descriptive message)
- [x] `tests/backend/test_ollama_client.py`

### `backend/core/model_registry.py`
- [x] Read `config/models.json`
- [x] `get_model_by_role(role)` → returns model for given role
- [x] `get_model_by_id(model_id)` → returns model details
- [x] `list_available_models()` → fetch from Ollama, match with config
- [x] `tests/backend/test_model_registry.py`

### `backend/core/memory_manager.py`
- [x] Create and read `skills.json`
- [x] Create and read `errors.json`
- [x] `log_error(error_type, context, solution)`
- [x] `resolve_error(error_id)`
- [x] `scan_errors(context)` → check for similar past errors
- [x] `update_skill_rate(skill_id, success: bool)`
- [x] `tests/backend/test_memory_manager.py`

### `backend/core/agent_base.py`
- [x] Abstract `AgentBase` class
- [x] Required methods: `get_capabilities()`, `run(task)`, `get_required_model_role()`
- [x] `ollama_client` and `memory_manager` integration
- [x] Pre-task `scan_errors()` loop
- [x] Post-task `update_skill_rate()` loop
- [x] `generate_training_sample(task, result)` → saves to `training_candidates.json`
- [x] `tests/backend/test_agent_base.py`

### `backend/core/agent_registry.py`
- [x] Auto-scan `agents/` directory
- [x] Read each `config.json`, load `enabled: true` agents
- [x] Dynamically import `agent.py`
- [x] `find_by_capability(capability)`
- [x] `find_by_input_type(file_type)`
- [x] `list_all()` → all active agents
- [x] `tests/backend/test_agent_registry.py`

### `backend/core/orchestrator.py`
- [x] `asyncio.Semaphore(max_concurrent)` concurrency control
- [x] Send task to DeepSeek → receive plan JSON
- [x] Resolve `depends_on` dependencies from plan
- [x] Run independent steps in parallel
- [x] Run dependent steps sequentially
- [x] Collect results → send to DeepSeek for final report
- [x] `tests/backend/test_orchestrator.py`

---

## Phase 2 — RAG Agent

> Goal: First real agent — load and query documents.
> Branch: `feature/rag-agent`

- [x] `agents/rag_agent/config.json`
- [x] `agents/rag_agent/memory/skills.json` initial values
- [x] `agents/rag_agent/memory/errors.json` initial values
- [x] `load_document(path)` → PDF, TXT, DOCX (LangChain)
- [x] `chunk_document(doc)`
- [x] `embed_chunks(chunks)` → via `nomic-embed-text`
- [x] `store_to_chroma(embeddings, chunks)`
- [x] `query(question)` → embed question, search ChromaDB
- [x] `generate_answer(question, context)` → via `qwen3:4b`
- [x] `AgentBase.run(task)` implementation
- [x] `tests/agents/test_rag_agent.py`

---

## Phase 3 — Adaptive Memory Integration

> Branch: `feature/adaptive-memory`

- [x] Connect error loop to RAG Agent
- [x] Connect skill loop to RAG Agent
- [x] Show memory stats in CLI
- [x] Integration tests

---

## Phase 4 — CLI

> Branch: `feature/cli`

- [ ] Welcome screen and system status summary
- [ ] `load <file>` → load document
- [ ] `ask <question>` → query
- [ ] `list docs` → loaded documents
- [ ] `status` → active agents and models
- [ ] `memory stats` → skill and error statistics
- [ ] `exit`

---

## Phase 5 — Coder Agent

> Branch: `feature/coder-agent`

- [ ] `agents/coder_agent/config.json`
- [ ] Code file reading and chunking (line + function based)
- [ ] Code embedding + ChromaDB integration
- [ ] Query via `qwen2.5-coder:3b`
- [ ] `AgentBase.run(task)` implementation
- [ ] Tests

---

## Phase 6 — Web UI

> Branch: `feature/web-ui`

- [ ] FastAPI backend (`frontend/api/`)
- [ ] REST endpoints (load doc, query, agent status)
- [ ] Minimal HTML/CSS/JS interface
- [ ] Add frontend service to Docker Compose
- [ ] Activate `ci-frontend.yml` pipeline

---

## Phase 7 — QLoRA Fine-tuning

> Goal: Fine-tune the orchestrator with user's work history and preferences.
> Prerequisite: `task_history.json` data from Phase 1 must have sufficient volume.

- [ ] Set up QLoRA fine-tune pipeline (`unsloth` or `trl`)
- [ ] Convert `task_history.json` to fine-tune format (instruction-input-output)
- [ ] LoRA adapter training (base model unchanged, adapter is separate file)
- [ ] Integrate adapter with Ollama (`Modelfile`)
- [ ] A/B test: base model vs fine-tuned adapter quality comparison
- [ ] Multi-tenant structure: separate adapter files per user
- [ ] Documentation: fine-tune process and retraining guide

---

## Phase 8 — Agent Training Data Generation

> Goal: Each agent produces structured logs of completed tasks as future fine-tune material.

- [ ] Add `generate_training_sample(task, result)` to `agent_base.py`
- [ ] Define output format (instruction / input / output / quality_score / approved)
- [ ] Save to `memory/training_candidates.json`
- [ ] Quality filter (critical — unfiltered loop produces bad data):
  - Initial: human approval via CLI `approve` command
  - Future: automated validation via larger model
- [ ] Move approved samples to `task_history.json` fine-tune queue
- [ ] Periodic fine-tune trigger (e.g. every 100 approved samples)

---

## Phase 9 — Continual Learning Loop

> Goal: Closed loop where the system improves through use.
> Note: Experimental phase — introduce to production carefully.

- [ ] Automated quality filter (small classifier or rule-based)
- [ ] Catastrophic forgetting protection (EWC or replay buffer)
- [ ] Pre/post fine-tune benchmark test
- [ ] Adapter versioning (rollback if quality drops)
- [ ] User notification: "model updated" with rollback option

---

## Documentation (Ongoing)

- [ ] Google Style docstrings for every module
- [ ] `docs/ARCHITECTURE.md`
- [ ] `docs/CONTRIBUTING.md`
- [ ] `CHANGELOG.md`
- [ ] `agents/rag_agent/README.md`

---

## Progress Summary

| Phase | Status | Done / Total |
|-------|--------|--------------|
| Phase 0 — Infrastructure | Done | 23 / 23 |
| Phase 1 — Core Layers | ⏳ Waiting | 0 / 35 |
| Phase 2 — RAG Agent | ⏳ Waiting | 0 / 11 |
| Phase 3 — Adaptive Memory | ⏳ Waiting | 0 / 4 |
| Phase 4 — CLI | ⏳ Waiting | 0 / 7 |
| Phase 5 — Coder Agent | ⏳ Waiting | 0 / 6 |
| Phase 6 — Web UI | ⏳ Waiting | 0 / 5 |
| Phase 7 — QLoRA Fine-tuning | ⏳ Long term | 0 / 7 |
| Phase 8 — Training Data Generation | ⏳ Long term | 0 / 6 |
| Phase 9 — Continual Learning | ⏳ Research | 0 / 5 |
| Documentation | 🔄 Ongoing | 0 / 5 |

---

*Last updated: Phase 0 in progress*
