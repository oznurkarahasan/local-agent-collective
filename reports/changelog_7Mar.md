feat 1: implement coder agent with CLI integration

- agents/coder_agent/agent.py: code loading, chunking, embedding, semantic search
- agents/coder_agent/config.json: capabilities and model config
- agents/coder_agent/memory/skills.json: 5 initial skills
- agents/coder_agent/memory/errors.json: 5 known error scenarios
- interface/cli.py: routes .py/.js/.ts/... to CoderAgent, docs to RagAgent
- real integration test passed: EN and TR queries working with qwen2.5-coder:3b
- unit tests + integration tests, all passing
