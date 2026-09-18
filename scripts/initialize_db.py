import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.storage.database import connect


if __name__ == "__main__":
    settings = get_settings()
    connect(settings.sqlite_path)
    print(f"Initialized database at {settings.sqlite_path}")
