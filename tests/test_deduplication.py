from app.memory.deduplication import looks_like_update


def test_update_detection():
    assert looks_like_update(
        "User has started using Java for my current backend project.",
        "User prefers Python for backend development.",
    )
