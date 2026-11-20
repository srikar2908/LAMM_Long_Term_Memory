from __future__ import annotations


def create_google_genai_client(api_key: str):
    """Create a Gemini Developer API client with bounded request timeouts."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            # Keep the Gemini Developer API version explicit across SDK releases.
            api_version="v1beta",
            timeout=15_000,
        ),
    )
