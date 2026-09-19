from __future__ import annotations

from app.memory.deduplication import lexical_similarity, looks_like_update


def test_lexical_similarity():
    s1 = "User prefers Python for backend development."
    s2 = "Python is the user's preferred backend language."
    sim = lexical_similarity(s1, s2)
    assert sim > 0.30

    s3 = "User loves hiking in national parks."
    assert lexical_similarity(s1, s3) == 0.0


def test_update_detection_requires_evidence():
    # 1. Clear temporal transition signal on shared domain topic -> UPDATE
    assert looks_like_update(
        new_text="User has started using Java for my current backend project.",
        old_text="User prefers Python for backend development.",
    )

    # 2. "switched to" signal on shared topic -> UPDATE
    assert looks_like_update(
        new_text="User has switched to PostgreSQL instead of MySQL.",
        old_text="User uses MySQL database for the project.",
    )

    # 3. Merely related statements without supersession signal -> NOT an update
    assert not looks_like_update(
        new_text="User is learning machine learning with PyTorch.",
        old_text="User prefers Python for backend development.",
    )
