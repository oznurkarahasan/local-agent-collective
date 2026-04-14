# Indis.ai -- Independent Intelligence

> A fully local, privacy-first, extensible multi-agent AI ecosystem.

[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://hub.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)

---

## What is this?

**Local AI Agent Collective** is an extensible multi-agent AI system that runs entirely on your own hardware — no data ever leaves your machine.

- 100% local, works fully offline
- Adding a new agent requires no code changes — just a folder and a config file
- Single command setup with Docker
- Built for privacy-sensitive use cases: legal, medical, engineering, research

---

## Monorepo Structure
```
local-agent-collective/
│
├── .github/
│   ├── workflows/
│   │   ├── ci-backend.yml       ← triggered on backend/ changes
│   │   ├── ci-agents.yml        ← triggered on agents/ changes
│   │   └── ci-frontend.yml      ← triggered on frontend/ changes (future)
│   └── pull_request_template.md
│
├── backend/
│   └── core/
│       ├── agent_base.py        ← abstract base class for all agents
│       ├── agent_registry.py    ← auto-discovers and manages agents
│       ├── model_registry.py    ← manages model configurations
│       ├── memory_manager.py    ← adaptive memory (errors.json + skills.json)
│       ├── ollama_client.py     ← async Ollama API wrapper (lazy load)
│       ├── orchestrator.py      ← plans and distributes tasks
│       └── platform_utils.py   ← cross-platform path and command management
│
├── agents/
│   ├── rag_agent/
│   │   ├── agent.py
│   │   ├── config.json
│   │   └── memory/
│   └── coder_agent/             ← future phase
│
├── config/
│   ├── models.json              ← all models and their roles (config, not code)
│   └── system.json              ← system settings (max_concurrent, keep_alive)
│
├── frontend/                    ← future phase (FastAPI + UI)
├── data/
│   ├── vector_store/            ← ChromaDB
│   └── documents/               ← user documents
│
├── infra/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   ├── backend/
│   ├── agents/
│   └── e2e/
│
├── docs/
├── interface/
│   └── cli.py
│
├── TODO.md
└── README.md
```

---

## Model Architecture

Models use **lazy loading**: a model is loaded only when needed and unloaded immediately after (`keep_alive: 0`). Concurrent execution is controlled via `asyncio.Semaphore`.

| Model | Role | Size | When Active |
|-------|------|------|-------------|
| `gemma4:e4b` | Orchestration, planning, CEO | ~5.0GB | Task analysis and final report |
| `deepseek-coder:7b` | Code analysis, generation, reasoning | ~4.5GB | Code file tasks |
| `phi3.5:moe` | Research, analysis, reasoning | ~4.0GB | General reasoning tasks |
| `llama3.2:3b` | RAG, documentation, doc_analysis | ~2.0GB | Text/PDF queries |
| `qwen2.5:1.5b` | QA, quality control, validation | ~1.2GB | Validation tasks |
| `nomic-embed-text:v1.5` | Embedding | ~274MB | Every document load |

### Task Flow
```
User submits task
        │
Gemma LOADS → analyzes task → produces plan JSON → UNLOADS
        │
        ├── Independent steps → run in parallel (Semaphore controlled)
        └── Dependent steps  → run sequentially (depends_on)
        │
Results collected
        │
Gemma LOADS → writes final report → UNLOADS → returned to user
```

---

## Plugin Architecture

The system uses a **config-driven agent registry**. No existing code changes needed to add a new agent:
```bash
mkdir agents/new_agent
touch agents/new_agent/agent.py
touch agents/new_agent/config.json
```

`config.json` example:
```json
{
  "id": "meeting_agent",
  "name": "Meeting Notes Agent",
  "capabilities": ["summarization", "meeting_notes"],
  "preferred_model_roles": ["reasoning"],
  "input_types": ["txt", "md"],
  "enabled": true
}
```

The registry auto-discovers the folder and the orchestrator uses it automatically.

| Change | Action Required |
|--------|----------------|
| Add new agent | New folder + config.json + agent.py |
| Add new model | 1 line in config/models.json |
| Disable an agent | Set `"enabled": false` in config.json |
| Change agent model | Update `preferred_model_roles` in config.json |

---

## System Requirements

### Minimum
| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | 4 cores | 8+ recommended |
| RAM | 16 GB | 8GB usable for models |
| VRAM | 4 GB | GTX 1650 / RX 580 equivalent |
| Storage | 20 GB free | Models take ~10GB |
| OS | Linux / macOS / Windows | Windows via Docker |

### Recommended
| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | 8+ cores | Faster parallel agent execution |
| RAM | 32 GB | Comfortable multi-model operation |
| VRAM | 8 GB+ | RTX 3070 / RX 6700 XT equivalent |
| Storage | 50 GB free | Room for multiple models |
| OS | Linux | Best Ollama performance |

### Model VRAM Requirements

| Model | VRAM (Q4) | Fits in 4GB | Fits in 8GB |
|-------|-----------|-------------|-------------|
| `gemma4:e4b` | ~5.0 GB | ❌ No | ✅ Yes |
| `deepseek-coder:7b` | ~4.5 GB | ⚠️ Tight | ✅ Yes |
| `phi3.5:moe` | ~4.0 GB | ⚠️ Tight | ✅ Yes |
| `llama3.2:3b` | ~2.0 GB | ✅ Yes | ✅ Yes |
| `qwen2.5:1.5b` | ~1.2 GB | ✅ Yes | ✅ Yes |
| `nomic-embed-text:v1.5` | ~274 MB | ✅ Yes | ✅ Yes |

> **Note:** Models use lazy loading — only the active model is loaded into VRAM at a time.
> On systems with less than 8GB VRAM, `max_concurrent` in `config/system.json` should be set to `1`.

---

## Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- Git

### Start
```bash
git clone https://github.com/oznurkarahasan/local-agent-collective.git
cd local-agent-collective
docker compose -f infra/docker-compose.yml up --build
```

This automatically:
- Starts the Ollama service
- Pulls required models
- Initializes ChromaDB
- Starts the agent system

### CLI
```bash
docker compose -f infra/docker-compose.yml exec app python interface/cli.py
```

---

## Testing
```bash
# All tests
docker compose -f infra/docker-compose.yml run --rm app pytest tests/ -v

# Backend only
pytest tests/backend/ -v

# Agents only
pytest tests/agents/ -v

# tests
sudo docker run --rm -v $(pwd):/app local-agent-collective:test pytest tests/ -v
```

---

## Git Workflow

No direct push to `main`. All development goes through feature branches via PR.
```
main           ← stable, protected
  └── dev      ← active development
        ├── feature/rag-agent
        ├── feature/model-registry
        ├── fix/ollama-connection
        └── chore/docker-setup
```

### Branch Naming

| Type | Format | Example |
|------|--------|---------|
| New feature | `feature/<name>` | `feature/rag-agent` |
| Bug fix | `fix/<name>` | `fix/chroma-path` |
| Infrastructure | `chore/<name>` | `chore/docker-setup` |
| Documentation | `docs/<name>` | `docs/architecture` |

### Commit Message Format (Conventional Commits)
```
feat: add RAG agent document loading
fix: resolve Ollama connection timeout
chore: update Docker Compose configuration
docs: add architecture diagram
test: add memory_manager unit tests
```

---

## CI/CD (GitHub Actions)

Each pipeline is triggered only when its relevant directory changes (`paths` filter):
```
backend/** changed  →  ci-backend.yml  →  lint + test + docker build
agents/**  changed  →  ci-agents.yml   →  lint + agent tests
```

Every PR requires: lint (flake8 + black) → unit tests (pytest) → docker build → minimum 1 review.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Model runtime | Ollama | Lazy loading, keep_alive=0 |
| Embedding | nomic-embed-text:v1.5 | Local, offline |
| Vector DB | ChromaDB | Pure Python, cross-platform |
| Document processing | LangChain | PDF, DOCX, TXT |
| Async | asyncio + Semaphore | Parallel agent control |
| Container | Docker + Compose | Single command setup |
| CI/CD | GitHub Actions | Path-based triggering |
| Path management | pathlib.Path | Cross-platform |

---

## Roadmap

See [TODO.md](TODO.md) for the full task list.

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Monorepo setup, Docker, GitHub Actions | done |
| 1 | Core layers (platform_utils → orchestrator) | done |
| 2 | RAG Agent + ChromaDB integration | done |
| 3 | Adaptive Memory integration | done |
| 4 | CLI + setup wizard | done |
| 5 | Coder Agent | done |
| 6 | Web UI (FastAPI + frontend) | done |
| 7 | QLoRA fine-tuning (personalized orchestrator) | ⏳ |
| 8 | Agent training data generation | ⏳ |
| 9 | Continual learning loop | ⏳ |

---

## WebUi 

```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn frontend.api.main:app --host 0.0.0.0 --port 8000 --reload

sudo docker run --rm -v $(pwd):/app local-agent-collective:test black frontend/api/
sudo docker run --rm -v $(pwd):/app local-agent-collective:test flake8 frontend/api/ --max-line-length=100
sudo docker run --rm -v $(pwd):/app local-agent-collective:test pytest tests/ -v
```

---

## Shortcuts

```bash
## Before every commit

# 1. Format code
sudo docker run --rm -v $(pwd):/app local-agent-collective:test black backend/ agents/

# 2. Check linting
sudo docker run --rm -v $(pwd):/app local-agent-collective:test flake8 backend/ agents/ --max-line-length=100

# 3. Run tests
sudo docker run --rm -v $(pwd):/app local-agent-collective:test pytest tests/ -v

# All three must pass before pushing.

## example: run before commit your current file
sudo docker run --rm -v $(pwd):/app local-agent-collective:test black backend/core/model_registry.py
sudo docker run --rm -v $(pwd):/app local-agent-collective:test flake8 backend/core/agent_base.py --max-line-length=100
```

## for the model downloading

```bash
+ ollama pull gemma4:e4b

+ ollama pull qwen2.5-coder:3b

+ ollama pull nomic-embed-text:v1.5
+ ollama pull llama3.2:3b

+ ollama pull deepseek-r1:1.5b
+ ollama pull phi3.5:latest

# start docker
sudo systemctl start docker

# to docker volume
docker exec -it local-agent-ollama ollama pull gemma4:e4b
docker exec -it local-agent-ollama ollama run gemma4:e4b

# run in web
PYTHONPATH=. .venv/bin/uvicorn frontend.api.main:app --host 0.0.0.0 --port 8000
```

> **"Your data stays with you."**
