"""
SMA — SMAClient (Public Interface)
====================================
Main entry point for the Sigma Memory Agent.

pip install sigma-memory-agent  (or: pip install -e .)

QUICK START:
    from sma import SMAClient

    sma = SMAClient(qwen_api_key="sk-...")
    memory_id = sma.remember({"content": "...", "domain": "trading"})
    memories  = sma.recall("XAU long setup")
    context   = sma.context_for("current market is trending up")

FULL USAGE with domain config:
    sma = SMAClient(
        qwen_api_key="sk-...",
        deepseek_api_key="sk-...",    # optional — enables audit
        db_path="sma_memory.db",
        regret_fn=lambda e: max(0, -e.get("outcome", {}).get("pnl_pct", 0) / 5.0),
        quality_rubric={
            "HIGH":       lambda e: e.get("outcome", {}).get("pnl_pct", 0) > 2.0,
            "FALSE_EDGE": lambda e: (e.get("outcome", {}).get("pnl_pct", 0) < 0
                                     and e.get("confidence", 0) > 0.7),
        },
    )
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sma.classifier import MemoryClassifier, QualityResult, RegretFn, QualityRubric
from sma.context_manager import ContextBlock, ContextManager
from sma.memory_store import MemoryRecord, MemoryStats, MemoryStore
from sma.retriever import EmbeddingClient, MemoryRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SMAConfig:
    """Configuration for SMAClient. All fields optional with sensible defaults."""

    db_path: str = "sma_memory.db"

    qwen_api_key: str = ""
    deepseek_api_key: str = ""          # empty = audit disabled

    # Retrieval
    default_recall_limit: int = 5
    min_similarity: float = 0.65
    token_budget: int = 2000

    # Classification
    temperature: float = 0.1
    audit_rubric: bool = False

    # Context management
    recency_weight: float = 0.2
    include_regret_score: bool = True

    # Embedding
    embed_on_store: bool = True         # embed immediately on remember()
    embed_async: bool = False           # future: async embedding

    @classmethod
    def from_env(cls) -> "SMAConfig":
        """Load config from environment variables."""
        return cls(
            qwen_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        )


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class SMAClient:
    """
    Sigma Memory Agent — persistent, queryable, quality-aware memory for AI agents.

    Domain-agnostic: works for trading, legal, creative, research, or any domain.
    Powered by Qwen Cloud (text-embedding-v4 + qwen3.7-plus) with optional
    DeepSeek audit for regret score cross-validation.
    """

    def __init__(
        self,
        qwen_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        db_path: Optional[str] = None,
        regret_fn: Optional[RegretFn] = None,
        quality_rubric: Optional[QualityRubric] = None,
        config: Optional[SMAConfig] = None,
    ):
        cfg = config or SMAConfig()
        cfg.qwen_api_key    = qwen_api_key    or cfg.qwen_api_key    or os.getenv("DASHSCOPE_API_KEY", "")
        cfg.deepseek_api_key = deepseek_api_key or cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        cfg.db_path = db_path or cfg.db_path
        self._cfg = cfg

        # Core components
        self._store = MemoryStore(db_path=cfg.db_path)

        self._embedder: Optional[EmbeddingClient] = None
        self._retriever: Optional[MemoryRetriever] = None
        if cfg.qwen_api_key:
            try:
                self._embedder = EmbeddingClient(api_key=cfg.qwen_api_key)
                self._retriever = MemoryRetriever(
                    store=self._store,
                    embedding_client=self._embedder,
                    default_limit=cfg.default_recall_limit,
                    min_similarity=cfg.min_similarity,
                )
            except Exception:
                pass  # Graceful degradation: store works without retrieval

        self._classifier: Optional[MemoryClassifier] = None
        if cfg.qwen_api_key:
            try:
                self._classifier = MemoryClassifier(
                    qwen_api_key=cfg.qwen_api_key,
                    deepseek_api_key=cfg.deepseek_api_key or None,
                    regret_fn=regret_fn,
                    quality_rubric=quality_rubric,
                    audit_rubric=cfg.audit_rubric,
                    temperature=cfg.temperature,
                )
            except Exception:
                pass

        self._ctx_mgr = ContextManager(
            token_budget=cfg.token_budget,
            recency_weight=cfg.recency_weight,
            include_regret_score=cfg.include_regret_score,
        )

    # ------------------------------------------------------------------
    # Core: remember
    # ------------------------------------------------------------------

    def remember(
        self,
        entry: Dict[str, Any],
        session_id: Optional[str] = None,
        auto_classify: bool = True,
    ) -> str:
        """
        Store a new memory. Returns memory_id.

        entry fields:
            content (str, required):      What happened / what was decided
            domain  (str, optional):      "trading", "legal", "creative", etc.
            outcome (dict, optional):     If provided AND auto_classify=True,
                                          triggers immediate quality classification
            metadata (dict, optional):    Any extra structured data
            confidence (float, optional): Agent's confidence in the decision
            decision_context (any, opt):  The context that led to this decision

        Returns:
            memory_id (str): Use this to update outcome later
        """
        content  = entry.get("content", str(entry))
        domain   = entry.get("domain", "general")
        outcome  = entry.get("outcome")
        metadata = entry.get("metadata", {})
        metadata.update({k: v for k, v in entry.items()
                         if k not in ("content", "domain", "outcome",
                                      "metadata", "embedding")})

        record = MemoryRecord.create(
            content=content,
            domain=domain,
            session_id=session_id or "default",
            metadata=metadata,
        )

        # Attach outcome if provided
        if outcome:
            record.outcome = outcome

        # Persist
        memory_id = self._store.insert(record)

        # Classify if outcome available
        if outcome and auto_classify and self._classifier:
            try:
                result = self._classifier.classify({**entry, "content": content})
                self._store.update_quality(
                    memory_id,
                    quality=result.classification,
                    regret_score=result.regret_score,
                    audit_consensus=result.consensus,
                    audit_disputed=result.disputed,
                )
                # Store classification metadata
                self._store.get(memory_id)  # refresh
            except Exception:
                pass  # Classification failure should not break storage

        # Embed for future retrieval
        if self._cfg.embed_on_store and self._retriever:
            record_fresh = self._store.get(memory_id)
            if record_fresh:
                try:
                    self._retriever.embed_and_store(record_fresh)
                except Exception:
                    pass  # Embedding failure is non-fatal

        return memory_id

    # ------------------------------------------------------------------
    # Core: update_outcome
    # ------------------------------------------------------------------

    def update_outcome(
        self,
        memory_id: str,
        outcome: Dict[str, Any],
        auto_classify: bool = True,
    ) -> Optional[QualityResult]:
        """
        Attach an outcome to an existing memory and optionally classify.
        Call this when a decision's result is known.

        Returns QualityResult if classification ran, else None.
        """
        self._store.update_outcome(memory_id, outcome)

        record = self._store.get(memory_id)
        if not record:
            return None

        # Re-embed with outcome (richer text)
        if self._retriever:
            try:
                self._retriever.embed_and_store(record)
            except Exception:
                pass

        if not auto_classify or not self._classifier:
            return None

        try:
            entry = {"content": record.content, "outcome": outcome, **record.metadata}
            result = self._classifier.classify(entry)
            self._store.update_quality(
                memory_id,
                quality=result.classification,
                regret_score=result.regret_score,
                audit_consensus=result.consensus,
                audit_disputed=result.disputed,
            )
            return result
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core: recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        domain: Optional[str] = None,
        min_quality: Optional[str] = None,
        limit: int = 5,
        as_context: bool = False,
        token_budget: Optional[int] = None,
    ) -> Any:
        """
        Retrieve memories semantically similar to query.

        Args:
            query:        Natural language query
            domain:       Restrict to domain (optional)
            min_quality:  Minimum quality level to include (optional)
            limit:        Max results
            as_context:   If True, return formatted ContextBlock instead of raw list
            token_budget: Per-call context token budget override (only used when
                         as_context=True). Safe to call concurrently — does not
                         mutate shared state.

        Returns:
            List[RetrievalResult] OR ContextBlock (if as_context=True)
        """
        if not self._retriever:
            # Fallback: recent memories without semantic search
            records = self._store.get_by_domain(domain or "", limit=limit,
                                                 min_quality=min_quality)
            if as_context:
                return self._ctx_mgr.build([], recent_memories=records,
                                           token_budget=token_budget)
            return []

        results = self._retriever.recall(
            query=query,
            domain=domain,
            min_quality=min_quality,
            limit=limit,
        )

        if as_context:
            recent = self._store.recent(limit=3, domain=domain)
            return self._ctx_mgr.build(results, recent_memories=recent,
                                       token_budget=token_budget)

        return results

    # ------------------------------------------------------------------
    # Core: context_for
    # ------------------------------------------------------------------

    def context_for(
        self,
        situation: str,
        domain: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Get a formatted context string ready to inject into an LLM prompt.
        This is the simplest way to use SMA — just call this before your LLM call.

        Returns formatted string with relevant memories + cautionary warnings.
        Empty string if no relevant memories found.

        Safe for concurrent use: max_tokens is passed through per-call rather
        than mutating shared ContextManager state.
        """
        block: ContextBlock = self.recall(
            situation, domain=domain, as_context=True, token_budget=max_tokens
        )
        parts = []
        if block.memories_text:
            parts.append(block.memories_text)
        if block.warnings_text:
            parts.append(block.warnings_text)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Proactive surfacing
    # ------------------------------------------------------------------

    def surface_related(
        self,
        new_entry: Dict[str, Any],
        limit: int = 3,
    ) -> str:
        """
        Before storing a new entry, surface related past memories proactively.
        Call this BEFORE remember() to give your agent relevant context first.

        Returns formatted proactive context string, or "" if nothing relevant.
        """
        if not self._retriever:
            return ""

        results = self._retriever.surface(
            new_entry=new_entry,
            domain=new_entry.get("domain"),
            limit=limit,
        )
        return self._ctx_mgr.build_proactive(results)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_session(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """Get all memories from a specific session."""
        return self._store.get_by_session(session_id, limit=limit)

    def get_high_regret(
        self,
        domain: Optional[str] = None,
        min_regret: float = 0.7,
        limit: int = 10,
    ) -> List[MemoryRecord]:
        """Get the most regret-heavy decisions — use for learning what to avoid."""
        return self._store.get_high_regret(domain=domain,
                                           min_regret=min_regret, limit=limit)

    def get_disputed(self, limit: int = 20) -> List[MemoryRecord]:
        """Get memories where Qwen and DeepSeek disagreed — for human review."""
        return self._store.get_disputed(limit=limit)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def classify(self, entry: Dict[str, Any]) -> Optional[QualityResult]:
        """Directly classify an entry without storing it."""
        if not self._classifier:
            return None
        return self._classifier.classify(entry)

    def stats(self, domain: Optional[str] = None) -> MemoryStats:
        """Get aggregate statistics for the memory store."""
        return self._store.stats(domain=domain)

    def reindex(self, limit: int = 100) -> int:
        """Batch-embed unindexed memories. Run periodically to keep search fresh."""
        if not self._retriever:
            return 0
        return self._retriever.embed_batch_unindexed(limit=limit)

    def pending_classification(self) -> int:
        """Number of memories with outcomes but not yet classified."""
        return len(self._store.get_unclassified_with_outcome(limit=1000))

    def run_pending_classification(self, limit: int = 50) -> int:
        """Classify all pending memories that have outcomes. Returns count classified."""
        if not self._classifier:
            return 0
        pending = self._store.get_unclassified_with_outcome(limit=limit)
        count = 0
        for record in pending:
            try:
                entry = {"content": record.content,
                         "outcome": record.outcome, **record.metadata}
                result = self._classifier.classify(entry)
                self._store.update_quality(
                    record.id,
                    quality=result.classification,
                    regret_score=result.regret_score,
                    audit_consensus=result.consensus,
                    audit_disputed=result.disputed,
                )
                count += 1
            except Exception:
                continue
        return count

    def __repr__(self) -> str:
        stats = self._store.stats()
        audit = "+" if self._cfg.deepseek_api_key else ""
        return (
            f"SMAClient(db={self._cfg.db_path!r}, "
            f"total={stats.total}, "
            f"models=Qwen{audit}DeepSeek)"
        )


# ---------------------------------------------------------------------------
# Self-test (offline — no API calls)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    print("=== SMAClient self-test (offline) ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create client without API keys — tests store + context logic only
        cfg = SMAConfig(
            db_path=os.path.join(tmpdir, "test.db"),
            qwen_api_key="",   # no API key = offline mode
            embed_on_store=False,
        )

        sma = SMAClient(config=cfg)
        print(f"Created: {sma}")

        # Test remember without outcome
        mid1 = sma.remember(
            {"content": "LONG XAUUSD at 3328.50", "domain": "trading",
             "confidence": 0.72},
            session_id="test_session",
        )
        print(f"Stored (no outcome): {mid1}")

        # Test remember with outcome (auto_classify=False since no API key)
        mid2 = sma.remember(
            {"content": "SHORT EURUSD at 1.0850", "domain": "trading",
             "outcome": {"pnl_pct": 1.5, "correct": True}, "confidence": 0.65},
            session_id="test_session",
            auto_classify=False,  # no API key
        )
        print(f"Stored (with outcome): {mid2}")

        # Test update_outcome
        sma.update_outcome(mid1, {"pnl_pct": 2.3, "correct": True},
                          auto_classify=False)
        record = sma._store.get(mid1)
        assert record.outcome == {"pnl_pct": 2.3, "correct": True}
        print(f"update_outcome: OK — outcome={record.outcome}")

        # Test get_session
        session_memories = sma.get_session("test_session")
        assert len(session_memories) == 2
        print(f"get_session: OK — {len(session_memories)} records")

        # Test stats
        stats = sma.stats(domain="trading")
        assert stats.total == 2
        print(f"stats: OK — total={stats.total}, by_quality={stats.by_quality}")

        # Test pending classification
        sma._store.update_outcome(mid1, {"pnl_pct": 2.3})
        pending = sma.pending_classification()
        print(f"pending_classification: {pending}")

        # Test context_for (no embedding, returns empty)
        ctx = sma.context_for("XAU long signal", domain="trading")
        print(f"context_for (no API): '{ctx[:50]}...' (empty expected)")

        print(f"\n{sma}")
        print("\nclient.py — OK")
