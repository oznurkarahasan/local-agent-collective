feat 1: Centralized Orchestration with Gemma 4 (CEO)

- backend/core/orchestrator.py: implemented planning and reporting flow using Gemma 4; added parallel/sequential step execution with dependency resolution
- backend/core/orchestrator.py: added robust JSON extraction handles (<think> tags, markdown blocks, nested braces)
- backend/core/orchestrator.py: implemented model warm-up strategy for embedding models (nomic-embed)

feat 2: Model Registry and Dynamic Resolution

- config/models.json: centralized model management by role (orchestration, rag, code_analysis, research, etc.)
- backend/core/ollama_client.py: added explicit keep_alive support for all model requests
- backend/core/agent_base.py: integrated ModelRegistry to allow agents to resolve preferred models dynamically
- config/system.json: system-wide configuration updates

feat 3: New Specialized Agents

- agents/qa_agent/agent.py: new agent for quality assurance and validation tasks
- agents/research_agent/agent.py: new agent for general knowledge, deep search, and synthesis
- agents/research_agent/config.json: research capabilities and model configuration

feat 4: Adaptive Memory and Infrastructure

- agents/*/memory/: removed fixed json files (errors, skills, training) in favor of system-driven adaptive learning
- backend/core/chroma_telemetry.py: disabled ChromaDB telemetry for improved privacy and performance
- frontend/api: updated routers and dependencies to suit the orchestrator-centric flow
- frontend/static/index.html: refreshed UI to support dynamic reporting from the orchestrator

Model Strategy Updates

- gemma4:e4b: designated as the "CEO" for orchestration and high-level planning
- deepseek-r1:1.5b: added for advanced reasoning, quality control, and validation (QA Agent)
- phi3.5:latest: integrated for broad research, synthesis, and web interpretation (Research Agent)
- qwen2.5-coder:3b: continues to handle code analysis and generation tasks
- llama3.2:3b: optimized for RAG, documentation, and translation tasks
- nomic-embed-text:v1.5: pinned in VRAM (keep_alive="-1") for constant availability during embedding tasks

Closes #24
