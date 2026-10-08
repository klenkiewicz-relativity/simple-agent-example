"""Append-only storage. A revision check protects the log, not external side effects."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol

Kind = Literal["input", "model", "tool"]


class Conflict(RuntimeError):
    """Another writer appended to this thread; do not blindly retry the run."""


@dataclass(frozen=True)
class Event:
    thread_id: str
    seq: int
    kind: Kind
    data: dict[str, Any]


class EventStore(Protocol):
    def read(self, thread_id: str) -> list[Event]: ...

    def append(self, thread_id: str, expected: int, kind: Kind, data: dict) -> Event:
        """Atomically append at expected + 1, or raise Conflict; JSON data only."""
        ...


def encode(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False)


class MemoryStore:
    """Same serialization and revision semantics as SQLite, without durability."""

    def __init__(self):
        self._events: dict[str, list[tuple[Kind, str]]] = {}
        self._lock = Lock()

    def read(self, thread_id: str) -> list[Event]:
        with self._lock:
            return [
                Event(thread_id, i, kind, json.loads(data))
                for i, (kind, data) in enumerate(self._events.get(thread_id, []), 1)
            ]

    def append(self, thread_id: str, expected: int, kind: Kind, data: dict) -> Event:
        payload = encode(data)
        with self._lock:
            events = self._events.setdefault(thread_id, [])
            if len(events) != expected:
                raise Conflict(thread_id)
            events.append((kind, payload))
        return Event(thread_id, expected + 1, kind, json.loads(payload))


class SQLiteStore:
    """Short transactions, one connection per operation; suitable for local persistence."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("Use MemoryStore for in-memory storage")
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "thread_id TEXT NOT NULL, seq INTEGER NOT NULL, "
                "kind TEXT NOT NULL, data TEXT NOT NULL, "
                "PRIMARY KEY (thread_id, seq))"
            )

    def _connect(self):
        # Connection context managers commit/rollback but do not close connections.
        # This tiny subclass gives each operation deterministic connection lifetime.
        class Connection(sqlite3.Connection):
            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    self.close()

        return sqlite3.connect(self.path, timeout=30, factory=Connection)

    def read(self, thread_id: str) -> list[Event]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT seq, kind, data FROM events WHERE thread_id=? ORDER BY seq", (thread_id,)
            ).fetchall()
        return [Event(thread_id, seq, kind, json.loads(data)) for seq, kind, data in rows]

    def append(self, thread_id: str, expected: int, kind: Kind, data: dict) -> Event:
        payload = encode(data)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE thread_id=?", (thread_id,)
            ).fetchone()[0]
            if version != expected:
                raise Conflict(thread_id)
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (thread_id, expected + 1, kind, payload),
            )
        return Event(thread_id, expected + 1, kind, json.loads(payload))
