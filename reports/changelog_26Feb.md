feat 1: add platform_utils with cross-platform support:

- get_base_dir() for Linux/macOS/Windows
- get_ollama_url() with env variable override
- is_ollama_running() via pgrep/tasklist
- get_ollama_install_instructions()
- get_vector_store_dir() and get_documents_dir()
- 8 unit tests, all passing"

feat 2: add ollama_client with async HTTP and retry logic

- ping() to check Ollama availability
- list_models() to fetch installed models
- pull_model() with streaming support
- chat() with keep_alive=0 default (lazy unload)
- embed() for vector generation
- unload_model() to free VRAM
- retry logic: configurable attempts and delay
- 12 unit tests, all passing"

feat 3: add model_registry with role and ID based lookup

- load models from config/models.json
- get_model_by_role() for role-based model selection
- get_model_by_id() for direct model lookup
- list_registered_models() for all configured models
- list_available_models() cross-references with Ollama
- ModelNotFoundError for missing models
- 11 unit tests, all passing"

feat 4: add memory_manager with adaptive skills and error tracking

- skills.json tracking with success rate calculation
- errors.json with log, resolve and scan functionality
- scan_errors() proactively finds known fixes before tasks
- training_candidates.json for future fine-tune data
- 20 unit tests, all passing"

feat 5: add agent_base with adaptive memory integration

- abstract AgentBase class with required interface
- run() with pre-task error scan and post-task skill update
- automatic error logging on failure
- automatic training sample generation on success
- get_agent_info() for agent status
- 16 unit tests, all passing"

feat 6: add agent_registry with auto-discovery

- auto-scans agents/ directory on initialization
- loads enabled agents from config.json
- dynamically imports agent.py classes
- find_by_capability() for capability-based lookup
- find_by_input_type() for file type routing
- get_class() returns importable agent class
- is_registered() for membership check
- 18 unit tests, all passing"