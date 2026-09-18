import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.llm.gemini import GeminiLLMClient


if __name__ == "__main__":
    settings = get_settings()
    if not settings.gemini_available:
        print("Gemini API key is not configured.")
        raise SystemExit(1)
    client = GeminiLLMClient(settings.gemini_api_key, settings.gemini_model)
    if not client.available:
        print(f"Gemini client unavailable: {client.last_error}")
        raise SystemExit(1)
    response = client.generate("Reply with exactly: LAMM Gemini smoke test passed")
    if client.last_error:
        print(f"Gemini call failed: {client.last_error}")
        raise SystemExit(1)
    print(response.strip())
