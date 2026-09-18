from datetime import datetime, timezone

import numpy as np

from app.memory.schemas import MemoryRecord
from app.storage.database import connect
from app.storage.repositories import MemoryRepository


def test_sqlite_persistence(tmp_path):
    conn = connect(tmp_path / "test.db")
    repo = MemoryRepository(conn, 3)
    now = datetime.now(timezone.utc)
    record = MemoryRecord(id="m1", text="User prefers Python.", created_at=now, updated_at=now)
    repo.create(record, np.array([1, 0, 0], dtype="float32"))
    assert repo.get("m1").text == "User prefers Python."
    assert repo.get_embedding("m1").shape == (3,)
