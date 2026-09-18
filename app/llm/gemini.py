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
        self.available = False
        self.last_error: str | None = None
        self.genai = None
        self.model_name = model_name
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(model_name)
            self.available = True
        except Exception as exc:
            self.model = None
            self.last_error = str(exc)
        self.fallback = MockLLMClient()

    def generate(self, prompt: str) -> str:
        if self.model is None:
            return self.fallback.generate(prompt)
        try:
            self.last_error = None
            return self.model.generate_content(prompt).text
        except Exception as exc:
            self.last_error = str(exc)
            fallback = self._build_available_fallback()
            if fallback is not None:
                try:
                    self.model = fallback
                    self.last_error = None
                    return self.model.generate_content(prompt).text
                except Exception as retry_exc:
                    self.last_error = str(retry_exc)
            return self.fallback.generate(prompt)

    def _build_available_fallback(self):
        if self.genai is None:
            return None
        preferred = [
            "models/gemini-flash-latest",
            "models/gemini-2.5-flash",
            "models/gemini-flash-lite-latest",
            "models/gemini-pro-latest",
        ]
        try:
            available = []
            for model in self.genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    available.append(model.name)
            selected = next((name for name in preferred if name in available), None)
            selected = selected or (available[0] if available else None)
            if selected:
                self.model_name = selected
                return self.genai.GenerativeModel(selected)
        except Exception as exc:
            self.last_error = str(exc)
        return None
