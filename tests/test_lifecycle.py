from app.core.config import Settings
from app.memory.lifecycle import LammLifecycleController
from app.memory.schemas import CandidateMemory, LifecycleOperation, MemoryRecord, RetrievedMemory
from datetime import datetime, timezone


def test_lifecycle_retain_without_redundancy():
    decision = LammLifecycleController(Settings()).decide(CandidateMemory(text="User prefers Python."), [], 0)
    assert decision.operation == LifecycleOperation.RETAIN


def test_lifecycle_compress_long_memory():
    text = "User said " + ("many details about an LLM memory project " * 20)
    decision = LammLifecycleController(Settings()).decide(CandidateMemory(text=text), [], 0)
    assert decision.operation == LifecycleOperation.COMPRESS


def test_lifecycle_merge_redundant_memory():
    now = datetime.now(timezone.utc)
    existing = MemoryRecord(id="m1", text="User prefers Python for backend development.", created_at=now, updated_at=now)
    similar = [RetrievedMemory(memory=existing, score=0.55)]
    decision = LammLifecycleController(Settings()).decide(CandidateMemory(text="Python is the user's preferred backend language."), similar, 1)
    assert decision.operation == LifecycleOperation.MERGE


def test_lifecycle_archive_low_confidence():
    decision = LammLifecycleController(Settings()).decide(CandidateMemory(text="Possible low-confidence memory: user browses AI news.", confidence_score=0.45), [], 0)
    assert decision.operation == LifecycleOperation.ARCHIVE


def test_lifecycle_forget_temporary():
    decision = LammLifecycleController(Settings()).decide(CandidateMemory(text="This is temporary and not very useful.", confidence_score=0.35), [], 0)
    assert decision.operation == LifecycleOperation.FORGET
