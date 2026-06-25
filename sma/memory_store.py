"""
SMA — Memory Store
==================
SQLite-backed persistent memory layer.
Thread-safe, domain-scoped, quality-aware.

Schema design principle: store everything, index what matters.
Quality classification is written AFTER outcome is known — never before.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """One unit of agent memory."""
    id: str
    content: str
    domain: str
    session_id: str
    timestamp: str
    quality: str                          # HIGH/MEDIUM/LOW/LUCKY/FALSE_EDGE/UNCLASSIFIED
    regret_score: float                   # 0.0 (no regret) → 1.0 (maximum regret)
    outcome: Optional[Dict[str, Any]]     # filled in later, triggers classification
    embedding: Optional[List[float]]      # text-embedding-v4 vector
    audit_consensus: bool                 # True if Qwen + DeepSeek agreed
    audit_disputed: bool                  # True if models disagreed (flag for human)
    metadata: Dict[str, Any]
    created_at: str

    @classmethod
    def create(
        cls,
        content: str,
        domain: str = "general",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemoryRecord":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            domain=domain,
            session_id=session_id or "default",
            timestamp=now,
            quality="UNCLASSIFIED",
            regret_score=-1.0,            # -1 = not yet scored
            outcome=None,
            embedding=None,
            audit_consensus=False,
            audit_disputed=False,
            metadata=metadata or {},
            created_at=now,
        )


@dataclass
class MemoryStats:
    """Aggregate statistics for a memory store."""
    total: int
    by_quality: Dict[str, int]
    by_domain: Dict[str, int]
    avg_regret_score: float
    disputed_count: int
    unclassified_count: int


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    SQLite-backed persistent memory store.

    Thread-safe via per-call connection acquisition.
    WAL mode for safe concurrent reads during writes.
    """

    # SQLite PRAGMA for local hardware performance
    _PRAGMAS = [
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA cache_size=-131072",   # 128 MB
        "PRAGMA mmap_size=268435456",  # 256 MB
        "PRAGMA temp_store=MEMORY",
    ]

    def __init__(self, db_path: str = "sma_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        for pragma in self._PRAGMAS:
            conn.execute(pragma)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id              TEXT PRIMARY KEY,
                    content         TEXT NOT NULL,
                    domain          TEXT NOT NULL DEFAULT 'general',
                    session_id      TEXT NOT NULL DEFAULT 'default',
                    timestamp       TEXT NOT NULL,
                    quality         TEXT NOT NULL DEFAULT 'UNCLASSIFIED',
                    regret_score    REAL NOT NULL DEFAULT -1.0,
                    outcome         TEXT,
                    embedding       TEXT,
                    audit_consensus INTEGER NOT NULL DEFAULT 0,
                    audit_disputed  INTEGER NOT NULL DEFAULT 0,
                    metadata        TEXT NOT NULL DEFAULT '{}',
                    created_at      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_domain
                    ON memories (domain);
                CREATE INDEX IF NOT EXISTS idx_memories_session
                    ON memories (session_id);
                CREATE INDEX IF NOT EXISTS idx_memories_quality
                    ON memories (quality);
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                    ON memories (timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_regret
                    ON memories (regret_score DESC);

                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    domain      TEXT NOT NULL DEFAULT 'general',
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT,
                    metadata    TEXT NOT NULL DEFAULT '{}'
                );
            """)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert(self, record: MemoryRecord) -> str:
        """Persist a new memory. Returns the record id."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, content, domain, session_id, timestamp, quality,
                     regret_score, outcome, embedding, audit_consensus,
                     audit_disputed, metadata, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.id, record.content, record.domain,
                    record.session_id, record.timestamp, record.quality,
                    record.regret_score,
                    json.dumps(record.outcome) if record.outcome else None,
                    json.dumps(record.embedding) if record.embedding else None,
                    int(record.audit_consensus),
                    int(record.audit_disputed),
                    json.dumps(record.metadata),
                    record.created_at,
                ),
            )
        return record.id

    def update_quality(
        self,
        memory_id: str,
        quality: str,
        regret_score: float,
        audit_consensus: bool = False,
        audit_disputed: bool = False,
    ) -> bool:
        """Write quality classification result. Called by classifier after outcome is known."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET quality=?, regret_score=?,
                    audit_consensus=?, audit_disputed=?
                WHERE id=?
                """,
                (quality, regret_score,
                 int(audit_consensus), int(audit_disputed),
                 memory_id),
            )
            return cursor.rowcount > 0

    def update_outcome(
        self,
        memory_id: str,
        outcome: Dict[str, Any],
    ) -> bool:
        """Attach an outcome to an existing memory. Triggers classification externally."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE memories SET outcome=? WHERE id=?",
                (json.dumps(outcome), memory_id),
            )
            return cursor.rowcount > 0

    def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """Store the embedding vector for this memory."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE memories SET embedding=? WHERE id=?",
                (json.dumps(embedding), memory_id),
            )
            return cursor.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_session(
        self,
        session_id: str,
        limit: int = 50,
        min_quality: Optional[str] = None,
    ) -> List[MemoryRecord]:
        quality_filter = self._quality_filter(min_quality)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE session_id=? {quality_filter}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_domain(
        self,
        domain: str,
        limit: int = 50,
        min_quality: Optional[str] = None,
        exclude_unclassified: bool = False,
    ) -> List[MemoryRecord]:
        quality_filter = self._quality_filter(min_quality)
        unclassified_filter = (
            "AND quality != 'UNCLASSIFIED'" if exclude_unclassified else ""
        )
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE domain=? {quality_filter} {unclassified_filter}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (domain, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_unclassified_with_outcome(self, limit: int = 100) -> List[MemoryRecord]:
        """Fetch memories that have outcomes but haven't been classified yet."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE quality='UNCLASSIFIED' AND outcome IS NOT NULL
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_disputed(self, limit: int = 50) -> List[MemoryRecord]:
        """Return memories where Qwen and DeepSeek disagreed — for human review."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE audit_disputed=1
                ORDER BY regret_score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_high_regret(
        self,
        domain: Optional[str] = None,
        min_regret: float = 0.7,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        """Surface the most regret-heavy decisions for learning."""
        domain_filter = f"AND domain='{domain}'" if domain else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE regret_score >= ? {domain_filter}
                ORDER BY regret_score DESC
                LIMIT ?
                """,
                (min_regret, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_with_embeddings(
        self,
        domain: Optional[str] = None,
        min_quality: Optional[str] = None,
        limit: int = 500,
    ) -> List[MemoryRecord]:
        """Return records that have embeddings, for similarity search."""
        domain_filter = f"AND domain='{domain}'" if domain else ""
        quality_filter = self._quality_filter(min_quality)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE embedding IS NOT NULL
                {domain_filter} {quality_filter}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def recent(self, limit: int = 10, domain: Optional[str] = None) -> List[MemoryRecord]:
        domain_filter = f"AND domain='{domain}'" if domain else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE 1=1 {domain_filter}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def stats(self, domain: Optional[str] = None) -> MemoryStats:
        domain_clause = f"domain='{domain}'" if domain else None

        def _where(*conditions) -> str:
            active = [c for c in conditions if c]
            return ("WHERE " + " AND ".join(active)) if active else ""

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM memories {_where(domain_clause)}"
            ).fetchone()[0]

            quality_rows = conn.execute(
                f"""
                SELECT quality, COUNT(*) as cnt
                FROM memories {_where(domain_clause)}
                GROUP BY quality
                """
            ).fetchall()
            by_quality = {r[0]: r[1] for r in quality_rows}

            domain_rows = conn.execute(
                f"""
                SELECT domain, COUNT(*) as cnt
                FROM memories {_where(domain_clause)}
                GROUP BY domain
                """
            ).fetchall()
            by_domain = {r[0]: r[1] for r in domain_rows}

            avg_regret = conn.execute(
                f"""
                SELECT AVG(regret_score)
                FROM memories {_where(domain_clause, 'regret_score >= 0')}
                """
            ).fetchone()[0] or 0.0

            disputed = conn.execute(
                f"""
                SELECT COUNT(*) FROM memories
                {_where(domain_clause, 'audit_disputed=1')}
                """
            ).fetchone()[0]

        return MemoryStats(
            total=total,
            by_quality=by_quality,
            by_domain=by_domain,
            avg_regret_score=round(avg_regret, 4),
            disputed_count=disputed,
            unclassified_count=by_quality.get("UNCLASSIFIED", 0),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quality_order() -> Dict[str, int]:
        return {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "LUCKY": 1, "FALSE_EDGE": 0}

    @staticmethod
    def _quality_filter(min_quality: Optional[str]) -> str:
        """Generate SQL fragment to filter by minimum quality level."""
        order = {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "LUCKY": 1, "FALSE_EDGE": 0}
        if not min_quality or min_quality not in order:
            return ""
        threshold = order[min_quality]
        allowed = [q for q, v in order.items() if v >= threshold]
        placeholders = ",".join(f"'{q}'" for q in allowed)
        return f"AND quality IN ({placeholders})"

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            domain=row["domain"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            quality=row["quality"],
            regret_score=row["regret_score"],
            outcome=json.loads(row["outcome"]) if row["outcome"] else None,
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            audit_consensus=bool(row["audit_consensus"]),
            audit_disputed=bool(row["audit_disputed"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))

        rec = MemoryRecord.create(
            content="Signal: LONG XAUUSD at 3328.50 with confidence 0.72",
            domain="trading",
            session_id="test_session",
            metadata={"signal_type": "mean_reversion", "tf": "H4"},
        )
        memory_id = store.insert(rec)
        print(f"Inserted: {memory_id}")

        store.update_outcome(memory_id, {"pnl_pct": 2.3, "correct": True})
        store.update_quality(memory_id, "HIGH", 0.05,
                             audit_consensus=True, audit_disputed=False)

        retrieved = store.get(memory_id)
        print(f"Quality:  {retrieved.quality}")
        print(f"Regret:   {retrieved.regret_score}")
        print(f"Outcome:  {retrieved.outcome}")

        stats = store.stats(domain="trading")
        print(f"Stats:    {stats}")
        print("memory_store.py — OK")
