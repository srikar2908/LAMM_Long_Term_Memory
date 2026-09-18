from __future__ import annotations


class MemoryCompressor:
    def __init__(self, max_length: int = 180):
        self.max_length = max_length

    def compress(self, text: str) -> str:
        clean = " ".join(text.split())
        if len(clean) <= self.max_length:
            return clean
        cut = clean[: self.max_length].rsplit(" ", 1)[0]
        return f"{cut}..."

    def merge(self, texts: list[str]) -> str:
        seen: list[str] = []
        lowered: set[str] = set()
        for text in texts:
            normalized = text.strip()
            if normalized and normalized.lower() not in lowered:
                seen.append(normalized)
                lowered.add(normalized.lower())
        return self.compress(" ".join(seen))
