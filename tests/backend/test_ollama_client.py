"""
Tests for ollama_client.py

Note: These tests use mocking — no real Ollama connection needed.
Real integration tests will be added when Docker Compose is fully set up.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.ollama_client import OllamaClient, OllamaConnectionError, OllamaModelError


@pytest.fixture
def client():
    """Create a test OllamaClient instance."""
    return OllamaClient(
        base_url="http://localhost:11434",
        retry_attempts=2,
        retry_delay=0.1,
    )


# --- ping ---

@pytest.mark.asyncio
async def test_ping_success(client):
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_http:
        mock_http.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = await client.ping()
        assert result is True


@pytest.mark.asyncio
async def test_ping_failure(client):
    with patch("httpx.AsyncClient") as mock_http:
        mock_http.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("connection refused")
        )
        result = await client.ping()
        assert result is False


# --- list_models ---

@pytest.mark.asyncio
async def test_list_models_success(client):
    mock_data = {"models": [{"name": "qwen3:4b"}, {"name": "deepseek-r1:7b"}]}

    with patch.object(client, "_request", new=AsyncMock(return_value=mock_data)):
        models = await client.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "qwen3:4b"


@pytest.mark.asyncio
async def test_list_models_empty(client):
    with patch.object(client, "_request", new=AsyncMock(return_value={"models": []})):
        models = await client.list_models()
        assert models == []


# --- chat ---

@pytest.mark.asyncio
async def test_chat_success(client):
    mock_data = {"message": {"role": "assistant", "content": "Hello!"}}

    with patch.object(client, "_request", new=AsyncMock(return_value=mock_data)):
        response = await client.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response == "Hello!"


@pytest.mark.asyncio
async def test_chat_keep_alive_default(client):
    mock_data = {"message": {"content": "response"}}

    with patch.object(client, "_request", new=AsyncMock(return_value=mock_data)) as mock_req:
        await client.chat(model="qwen3:4b", messages=[])
        call_args = mock_req.call_args
        assert call_args[0][2]["keep_alive"] == "0"


# --- embed ---

@pytest.mark.asyncio
async def test_embed_success_single(client):
    mock_data = {"embeddings": [[0.1, 0.2, 0.3]]}

    with patch.object(client, "_request", new=AsyncMock(return_value=mock_data)):
        vector = await client.embed(model="nomic-embed-text:v1.5", text="hello")
        assert vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_success_batch(client):
    mock_data = {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}

    with patch.object(client, "_request", new=AsyncMock(return_value=mock_data)):
        vectors = await client.embed(model="nomic-embed-text:v1.5", text=["hello", "world"])
        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]
        assert vectors[1] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_embed_empty_response_raises(client):
    with patch.object(client, "_request", new=AsyncMock(return_value={"embeddings": []})):
        with pytest.raises(OllamaModelError):
            await client.embed(model="nomic-embed-text:v1.5", text="hello")


# --- unload_model ---

@pytest.mark.asyncio
async def test_unload_model_success(client):
    with patch.object(client, "_request", new=AsyncMock(return_value={})):
        result = await client.unload_model("qwen3:4b")
        assert result is True


@pytest.mark.asyncio
async def test_unload_model_failure_returns_false(client):
    with patch.object(client, "_request", new=AsyncMock(side_effect=Exception("error"))):
        result = await client.unload_model("qwen3:4b")
        assert result is False


# --- _request retry logic ---

@pytest.mark.asyncio
async def test_request_retries_on_connection_error(client):
    import httpx

    with patch("httpx.AsyncClient") as mock_http:
        mock_http.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(OllamaConnectionError):
            await client._request("GET", "/api/tags")


@pytest.mark.asyncio
async def test_request_raises_model_error_on_http_status(client):
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model not found"

    with patch("httpx.AsyncClient") as mock_http:
        mock_http.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "not found", request=MagicMock(), response=mock_response
            )
        )
        with pytest.raises(OllamaModelError):
            await client._request("POST", "/api/chat", {})
