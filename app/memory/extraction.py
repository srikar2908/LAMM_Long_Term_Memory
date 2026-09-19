from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from pydantic import ValidationError

from app.memory.schemas import CandidateMemory


class MemoryExtractor(ABC):
    @abstractmethod
    def extract(self, user_message: str, assistant_message: str | None = None) -> list[CandidateMemory]:
        pass


class DeterministicMemoryExtractor(MemoryExtractor):
    """Rule-based extractor that keeps tests and demos reproducible."""

    patterns = [
        (re.compile(r"\bi prefer ([^.]+)", re.I), "User prefers {value}.", 0.92),
        (re.compile(r"\bi am working on ([^.]+)", re.I), "User is working on {value}.", 0.88),
        (re.compile(r"\bmy project uses ([^.]+)", re.I), "User's project uses {value}.", 0.86),
        (re.compile(r"\bi also like ([^.]+)", re.I), "User likes {value}.", 0.78),
        (re.compile(r"\bi have started using ([^.]+)", re.I), "User has started using {value}.", 0.90),
        (re.compile(r"\bpython is my preferred ([^.]+)", re.I), "Python is the user's preferred {value}.", 0.88),
        (re.compile(r"\bmy current backend project uses ([^.]+)", re.I), "User's current backend project uses {value}.", 0.91),
        (re.compile(r"\bmaybe remember that ([^.]+)", re.I), "Possible low-confidence memory: {value}.", 0.45),
    ]

    def extract(self, user_message: str, assistant_message: str | None = None) -> list[CandidateMemory]:
        candidates: list[CandidateMemory] = []
        for pattern, template, confidence in self.patterns:
            for match in pattern.finditer(user_message):
                value = match.group(1).strip().rstrip(".")
                candidates.append(
                    CandidateMemory(
                        text=template.format(value=value),
                        confidence_score=confidence,
                        source="deterministic",
                        metadata={"pattern": pattern.pattern},
                    )
                )
        if not candidates and user_message.strip():
            msg_lower = user_message.lower()
            is_ephemeral = "temporary" in msg_lower or "not very useful" in msg_lower or "filler" in msg_lower
            confidence = 0.35 if is_ephemeral else (0.60 if len(user_message.split()) > 10 else 0.50)
            candidates.append(
                CandidateMemory(
                    text=user_message.strip(),
                    confidence_score=confidence,
                    source="deterministic_fallback",
                    metadata={"is_ephemeral": is_ephemeral},
                )
            )
        return candidates


class GeminiMemoryExtractor(MemoryExtractor):
    def __init__(self, api_key: str, model_name: str, fallback: MemoryExtractor | None = None):
        self.fallback = fallback or DeterministicMemoryExtractor()
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception:
            self.model = None

    def extract(self, user_message: str, assistant_message: str | None = None) -> list[CandidateMemory]:
        if self.model is None:
            return self.fallback.extract(user_message, assistant_message)
        prompt = (
            "Extract durable long-term memory candidates from the user message. "
            "Return JSON list of objects with text, confidence_score, source, metadata. "
            "Do not include secrets or short-lived conversational filler.\n"
            f"User: {user_message}\nAssistant: {assistant_message or ''}"
        )
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            raw = json.loads(text)
            return [CandidateMemory.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValidationError, Exception):
            return self.fallback.extract(user_message, assistant_message)
