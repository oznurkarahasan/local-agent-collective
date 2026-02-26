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
