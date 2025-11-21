import json
import logging

import requests

logger = logging.getLogger(__name__)


class LocalGenerator:
    """
    Generate an answer using local LLM (via Ollama API).
    """

    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """
        Generate an answer.
        """
        if not context_chunks:
            return "No relevant context found."

        context = "\n\n".join(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer concisely:"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()  # raise for non 200 status
            data = response.json()
            return data.get("response", "").strip()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decode failed: {e}. Raw response: {response.text[:1000]}"
            )
            raise RuntimeError(f"Invalid JSON response from Ollama: {e}") from e
