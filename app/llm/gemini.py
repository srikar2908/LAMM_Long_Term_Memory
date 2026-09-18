from __future__ import annotations

from app.llm.base import LLMClient


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
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception:
            self.model = None
        self.fallback = MockLLMClient()

    def generate(self, prompt: str) -> str:
        if self.model is None:
            return self.fallback.generate(prompt)
        try:
            return self.model.generate_content(prompt).text
        except Exception:
            return self.fallback.generate(prompt)
