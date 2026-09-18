from fastapi import FastAPI

from app.api import routes_chat, routes_evaluation, routes_memory
from app.core.logging_config import configure_logging

configure_logging()

app = FastAPI(title="LAMM Long-Term Memory API", version="0.1.0")
app.include_router(routes_chat.router)
app.include_router(routes_memory.router)
app.include_router(routes_evaluation.router)


@app.get("/")
def root():
    return {"name": "LAMM", "status": "running", "mode": "gemini_optional"}
