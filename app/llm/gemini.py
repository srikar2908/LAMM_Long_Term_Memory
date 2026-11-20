from __future__ import annotations

import logging

from app.llm.base import LLMClient
from app.llm.google_genai_client import create_google_genai_client

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    def generate(self, prompt: str) -> str:
        lower = prompt.lower()
        if "what programming language" in lower and "java" in lower:
            return "You are currently using Java for your backend project."
        if "what programming language" in lower and "python" in lower:
            return "Your stored memories mention Python, but I do not see a newer current-language update."
        return "Mock response generated from the retrieved memory context."


class GeminiLLMClient(LLMClient):
    def __init__(self, api_key: str, model_name: str):
        self.available = False
        self.last_error: str | None = None
        self.model_name = model_name
        self.client = None
        # ContextBuilder uses the model object only when it exposes count_tokens.
        # Token counts remain estimated for the new SDK's simple prompt adapter.
        self.model = None
        try:
            self.client = create_google_genai_client(api_key)
            self.model = model_name
            self.available = True
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Gemini client is unavailable; using offline mock: %s", exc)
        self.fallback = MockLLMClient()

    def generate(self, prompt: str) -> str:
        if self.client is None:
            return self.fallback.generate(prompt)
        try:
            self.last_error = None
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if response.text:
                return response.text
            raise ValueError("Gemini returned an empty text response.")
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Gemini generation failed; using offline mock: %s", exc)
            return self.fallback.generate(prompt)
