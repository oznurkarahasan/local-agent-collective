"""
ollama_client.py

Async HTTP client for communicating with the local Ollama API.
All model interactions go through this module.
"""

import json
import asyncio
import httpx
from typing import Optional
from backend.core.platform_utils import PlatformUtils


class OllamaConnectionError(Exception):
    """Raised when Ollama service is unreachable."""

    pass


class OllamaModelError(Exception):
    """Raised when a model operation fails."""

    pass


class OllamaClient:
    """Async client for the Ollama API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
    ):
        self.base_url = base_url or PlatformUtils.get_ollama_url()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout

    async def ping(self) -> bool:
        """
        Check if Ollama service is reachable.

        Returns:
            True if reachable, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict]:
        """
        List all locally installed models.

        Returns:
            List of model info dicts.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
        """
        response = await self._request("GET", "/api/tags")
        return response.get("models", [])

    async def pull_model(self, model_id: str) -> bool:
        """
        Pull (download) a model from Ollama registry.

        Args:
            model_id: Model identifier e.g. 'qwen3:4b'

        Returns:
            True if successful.

        Raises:
            OllamaModelError: If pull fails.
        """
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_id},
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "error" in data:
                                raise OllamaModelError(
                                    f"Failed to pull {model_id}: {data['error']}"
                                )
                            if status == "success":
                                return True
            return True
        except OllamaModelError:
            raise
        except Exception as e:
            raise OllamaModelError(f"Failed to pull {model_id}: {e}") from e

    async def chat(
        self,
        model: str,
        messages: list[dict],
        keep_alive: int | str = 0,
        stream: bool = False,
    ) -> str:
        """
        Send a chat completion request to Ollama.

        Args:
            model: Model identifier e.g. 'qwen3:4b'
            messages: List of message dicts with 'role' and 'content'
            keep_alive: How long to keep model in memory. '0' = unload immediately
            stream: Whether to stream the response

        Returns:
            Model response as string.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
            OllamaModelError: If model is not available.
        """
        payload = {
            "model": model,
            "messages": messages,
            "keep_alive": keep_alive,
            "stream": stream,
        }
        response = await self._request("POST", "/api/chat", payload)
        return response.get("message", {}).get("content", "")

    async def embed(
        self,
        model: str,
        text: str,
        keep_alive: int | str | None = None,
    ) -> list[float]:
        """
        Generate embeddings for the given text.

        Args:
            model: Embedding model e.g. 'nomic-embed-text:v1.5'
            text: Text to embed
            keep_alive: How long to keep model in memory.

        Returns:
            Embedding vector as list of floats.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
            OllamaModelError: If embedding fails.
        """
        payload = {"model": model, "input": text}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        response = await self._request("POST", "/api/embed", payload)
        embeddings = response.get("embeddings", [])
        if not embeddings:
            raise OllamaModelError(f"No embeddings returned for model {model}")
        return embeddings[0]

    async def unload_model(self, model: str) -> bool:
        """
        Unload a model from memory immediately.

        Args:
            model: Model identifier to unload

        Returns:
            True if successful.
        """
        try:
            await self._request(
                "POST",
                "/api/chat",
                {
                    "model": model,
                    "messages": [],
                    "keep_alive": 0,
                },
            )
            return True
        except Exception:
            return False

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[dict] = None,
    ) -> dict:
        """
        Internal method to make HTTP requests with retry logic.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            payload: Request body

        Returns:
            Response as dict.

        Raises:
            OllamaConnectionError: After all retry attempts fail.
        """
        url = f"{self.base_url}{endpoint}"
        last_error = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        response = await client.get(url)
                    else:
                        response = await client.post(url, json=payload)

                    response.raise_for_status()
                    return response.json()

            except httpx.ConnectError as e:
                last_error = e
                if attempt < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay)

            except httpx.HTTPStatusError as e:
                raise OllamaModelError(
                    f"Ollama API error {e.response.status_code}: {e.response.text}"
                ) from e

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay)

        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {self.base_url} "
            f"after {self.retry_attempts} attempts. "
            f"Is Ollama running? Run: {PlatformUtils.get_ollama_install_instructions()}"
        ) from last_error
