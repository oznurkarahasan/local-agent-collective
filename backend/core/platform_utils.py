"""
platform_utils.py

Cross-platform path and command management.
All platform-specific logic is centralized here.
No other module should contain platform checks.
"""

import platform
import os
import subprocess
from pathlib import Path


class PlatformUtils:
    """Handles all platform-specific operations."""

    OS = platform.system()  # "Linux" | "Darwin" | "Windows"

    @staticmethod
    def get_base_dir() -> Path:
        """
        Returns the base data directory for the application.

        Linux  : ~/.local/share/local-agent-collective/
        macOS  : ~/Library/Application Support/local-agent-collective/
        Windows: C:/Users/<user>/AppData/Local/local-agent-collective/
        """
        os_name = platform.system()
        home = Path.home()

        if os_name == "Linux":
            base = home / ".local" / "share" / "local-agent-collective"
        elif os_name == "Darwin":
            base = home / "Library" / "Application Support" / "local-agent-collective"
        elif os_name == "Windows":
            app_data = os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))
            base = Path(app_data) / "local-agent-collective"
        else:
            base = home / "local-agent-collective"

        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def get_ollama_url() -> str:
        """
        Returns the Ollama API base URL.
        Checks for environment variable override first (useful for Docker).
        """
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @staticmethod
    def is_ollama_running() -> bool:
        """
        Checks whether the Ollama service is currently running.

        Linux/macOS : checks via ps
        Windows     : checks via tasklist
        """
        os_name = platform.system()

        try:
            if os_name == "Windows":
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq ollama.exe"],
                    capture_output=True,
                    text=True,
                )
                return "ollama.exe" in result.stdout
            else:
                result = subprocess.run(
                    ["pgrep", "-x", "ollama"],
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def get_ollama_install_instructions() -> str:
        """
        Returns platform-specific Ollama installation instructions.
        """
        os_name = platform.system()

        instructions = {
            "Linux": "curl -fsSL https://ollama.ai/install.sh | sh",
            "Darwin": "brew install ollama  or  https://ollama.ai/download",
            "Windows": "Download installer from https://ollama.ai/download",
        }

        return instructions.get(os_name, "Visit https://ollama.ai/download")

    @staticmethod
    def get_vector_store_dir() -> Path:
        """Returns the ChromaDB vector store directory."""
        return PlatformUtils.get_base_dir() / "vector_store"

    @staticmethod
    def get_documents_dir() -> Path:
        """Returns the user documents directory."""
        return PlatformUtils.get_base_dir() / "documents"
