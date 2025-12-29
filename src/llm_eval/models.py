"""LLM model interface for Ollama."""

import subprocess
import time
from dataclasses import dataclass

import requests


@dataclass
class ModelInfo:
    """Information about an available model."""

    name: str
    size_bytes: int
    size_human: str
    modified: str

    @property
    def is_embedding_model(self) -> bool:
        """Check if this is an embedding-only model."""
        embedding_indicators = ["embed", "embedding"]
        return any(ind in self.name.lower() for ind in embedding_indicators)

    @property
    def is_text_model(self) -> bool:
        """Check if this is a text generation model."""
        return not self.is_embedding_model


def list_available_models() -> list[ModelInfo]:
    """List all models available in Ollama."""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    models = []
    lines = result.stdout.strip().split("\n")[1:]  # Skip header

    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            # Size is typically in format "1.6 GB" or "639 MB"
            size_str = f"{parts[2]} {parts[3]}"
            # Parse size to bytes (approximate)
            size_val = float(parts[2])
            if "GB" in parts[3]:
                size_bytes = int(size_val * 1024 * 1024 * 1024)
            elif "MB" in parts[3]:
                size_bytes = int(size_val * 1024 * 1024)
            else:
                size_bytes = int(size_val)

            modified = " ".join(parts[4:]) if len(parts) > 4 else ""

            models.append(
                ModelInfo(
                    name=name,
                    size_bytes=size_bytes,
                    size_human=size_str,
                    modified=modified,
                )
            )

    return models


def get_text_models() -> list[ModelInfo]:
    """Get only text generation models (exclude embedding models)."""
    return [m for m in list_available_models() if m.is_text_model]


class OllamaModel:
    """Interface to an Ollama model."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """Generate a response from the model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, latency_seconds)
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        response = requests.post(url, json=payload, timeout=self.timeout)
        latency = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        return result.get("response", ""), latency

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """Chat-style generation with message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, latency_seconds)
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start_time = time.time()
        response = requests.post(url, json=payload, timeout=self.timeout)
        latency = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        return result.get("message", {}).get("content", ""), latency

    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name") == self.model_name for m in models)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"OllamaModel({self.model_name!r})"
