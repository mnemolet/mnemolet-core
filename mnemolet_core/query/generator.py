import json
import requests
from mnemolet_core.config import OLLAMA_URL


class LocalGenerator:
    """
    Generate an answer using local LLM (via Ollama API).
    """

    def __init__(self, model: str = "llama3", host: str = OLLAMA_URL):
        self.model = model
        self.host = host

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
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            data = response.json()
            return data.get("response", "").strip()
        except json.JSONDecodeError as e:
            print(f"JSON decode failed {e}, raw Ollama output:")
            url = "http://localhost:11434/api/generate"
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            r = requests.post(url, json=payload)
            print(r.text[:1000])
            raise
