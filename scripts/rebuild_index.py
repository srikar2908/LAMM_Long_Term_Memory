import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.manager import MemoryManager


if __name__ == "__main__":
    manager = MemoryManager()
    manager.rebuild_index()
    print("Rebuilt vector index.")
    print(manager.stats())
