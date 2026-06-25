"""
SMA — Memory Retriever
======================
Semantic search over stored memories using Qwen text-embedding-v4.

Two retrieval modes:
  REACTIVE:   recall(query)  → find memories relevant to a given query
  PROACTIVE:  surface(entry) → find memories relevant to a NEW entry
              before it's stored (proactive surfacing)

Storage strategy:
  Embeddings stored as JSON in SQLite (no FAISS dependency).
  For hackathon scale (<100K memories), numpy cosine similarity
  is fast enough (<50ms for 10K memories).
  For production scale, swap _cosine_search() to pgvector/FAISS.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from sma.memory_store import MemoryRecord, MemoryStore


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    record: MemoryRecord
    similarity: float           # cosine similarity 0.0 → 1.0
    relevance_note: str         # why this memory was surfaced


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """
    Wrapper around Qwen text-embedding-v4.
    OpenAI-compatible endpoint — same SDK, different base_url.
    """

    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    MODEL    = "text-embedding-v4"

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not key:
            raise ValueError("DASHSCOPE_API_KEY required for embedding")
        self._client = OpenAI(api_key=key, base_url=self.BASE_URL)

    def embed(self, text: str) -> List[float]:
        """Embed a single text string. Returns float list."""
        resp = self._client.embeddings.create(
            model=self.MODEL,
            input=text,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in one API call (more efficient)."""
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self.MODEL,
            input=texts,
        )
        # Sort by index to preserve order
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MemoryRetriever:
    """
    Semantic similarity retrieval over the memory store.

    Workflow:
      1. Embed the query with text-embedding-v4
      2. Load candidate memories that already have embeddings
      3. Cosine similarity → rank → filter → return top-k
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_client: EmbeddingClient,
        default_limit: int = 5,
        min_similarity: float = 0.65,    # below this → not relevant enough
    ):
        self.store = store
        self.embedder = embedding_client
        self.default_limit = default_limit
        self.min_similarity = min_similarity

    # ------------------------------------------------------------------
    # Reactive retrieval
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        domain: Optional[str] = None,
        min_quality: Optional[str] = None,
        limit: Optional[int] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Find memories semantically similar to query.

        Args:
            query:       Natural language query or context string
            domain:      Restrict search to this domain
            min_quality: Only return memories at or above this quality level
            limit:       Max results (default: self.default_limit)
            exclude_ids: Memory IDs to exclude (e.g. ones already in context)
        Returns:
            List of RetrievalResult, ordered by similarity descending.
        """
        limit = limit or self.default_limit

        # Embed the query
        try:
            query_vec = np.array(self.embedder.embed(query), dtype=np.float32)
        except Exception as e:
            return []  # Fail gracefully — retrieval is non-critical

        # Load candidate memories
        candidates = self.store.get_with_embeddings(
            domain=domain,
            min_quality=min_quality,
            limit=500,   # pre-filter pool; cosine is fast at this scale
        )
        if not candidates:
            return []

        exclude_ids = set(exclude_ids or [])
        candidates = [c for c in candidates if c.id not in exclude_ids]

        return self._cosine_search(query_vec, candidates, limit)

    # ------------------------------------------------------------------
    # Proactive surfacing
    # ------------------------------------------------------------------

    def surface(
        self,
        new_entry: Dict[str, Any],
        domain: Optional[str] = None,
        min_quality: Optional[str] = "MEDIUM",
        limit: int = 3,
    ) -> List[RetrievalResult]:
        """
        Proactively find memories relevant to a NEW entry BEFORE it's stored.
        Called by client.remember() to return related past experiences.

        This is the "agent automatically recalls relevant past context" feature
        that makes SMA different from a simple store-and-retrieve system.
        """
        content = new_entry.get("content", "")
        if not content:
            return []

        # Build a rich query from the new entry
        query_parts = [content]
        if "domain" in new_entry:
            query_parts.append(f"domain: {new_entry['domain']}")
        if "decision_context" in new_entry:
            ctx = new_entry["decision_context"]
            if isinstance(ctx, dict):
                query_parts.append(json.dumps(ctx, default=str)[:300])
            else:
                query_parts.append(str(ctx)[:300])

        query = " | ".join(query_parts)
        domain = domain or new_entry.get("domain")

        results = self.recall(
            query=query,
            domain=domain,
            min_quality=min_quality,
            limit=limit,
        )

        # Tag as proactively surfaced
        for r in results:
            r.relevance_note = f"proactive: {r.relevance_note}"

        return results

    # ------------------------------------------------------------------
    # Embedding maintenance
    # ------------------------------------------------------------------

    def embed_and_store(self, record: MemoryRecord) -> bool:
        """
        Compute and persist embedding for a single memory.
        Called after insert to make the memory searchable.
        """
        try:
            embedding = self.embedder.embed(self._record_to_text(record))
            return self.store.update_embedding(record.id, embedding)
        except Exception:
            return False  # Non-fatal: memory is stored, just not searchable yet

    def embed_batch_unindexed(self, limit: int = 50) -> int:
        """
        Batch-embed memories that don't have embeddings yet.
        Run periodically (e.g. via DevOps scheduler) to keep search index fresh.
        Returns number of records indexed.
        """
        with_embeddings_ids = {
            r.id for r in self.store.get_with_embeddings(limit=10000)
        }
        # Get recent records without embeddings
        all_recent = self.store.recent(limit=limit + len(with_embeddings_ids))
        unindexed = [r for r in all_recent if r.id not in with_embeddings_ids][:limit]

        if not unindexed:
            return 0

        texts = [self._record_to_text(r) for r in unindexed]
        try:
            embeddings = self.embedder.embed_batch(texts)
            count = 0
            for record, emb in zip(unindexed, embeddings):
                if self.store.update_embedding(record.id, emb):
                    count += 1
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_search(
        self,
        query_vec: np.ndarray,
        candidates: List[MemoryRecord],
        limit: int,
    ) -> List[RetrievalResult]:
        """Batch cosine similarity. Pure numpy — no external dependencies."""
        if not candidates:
            return []

        # Build matrix: rows = candidates, cols = embedding dims
        matrix = np.array(
            [r.embedding for r in candidates if r.embedding],
            dtype=np.float32,
        )
        valid_candidates = [r for r in candidates if r.embedding]

        if matrix.shape[0] == 0:
            return []

        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_normalized = matrix / matrix_norms

        # Cosine similarities
        similarities = matrix_normalized @ query_norm

        # Filter by min_similarity and sort
        results = []
        for idx, sim in enumerate(similarities):
            if float(sim) >= self.min_similarity:
                record = valid_candidates[idx]
                results.append(RetrievalResult(
                    record=record,
                    similarity=round(float(sim), 4),
                    relevance_note=self._relevance_note(record, float(sim)),
                ))

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    @staticmethod
    def _relevance_note(record: MemoryRecord, similarity: float) -> str:
        quality_label = record.quality
        if similarity > 0.9:
            return f"highly_similar:{quality_label}"
        elif similarity > 0.75:
            return f"similar:{quality_label}"
        else:
            return f"related:{quality_label}"

    @staticmethod
    def _record_to_text(record: MemoryRecord) -> str:
        """Convert a memory record to a text representation for embedding."""
        parts = [record.content]
        if record.domain and record.domain != "general":
            parts.append(f"domain:{record.domain}")
        if record.outcome:
            parts.append(f"outcome:{json.dumps(record.outcome, default=str)[:200]}")
        if record.metadata:
            relevant_meta = {k: v for k, v in record.metadata.items()
                            if k not in ("embedding", "raw_response")}
            if relevant_meta:
                parts.append(f"meta:{json.dumps(relevant_meta, default=str)[:200]}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Self-test (offline — tests similarity math without API)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os, sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from sma.memory_store import MemoryRecord, MemoryStore

    print("=== Retriever self-test (offline cosine math) ===\n")

    # Test cosine similarity logic directly
    rng = np.random.default_rng(42)

    # Create 3 fake records with embeddings
    records = []
    for i, (label, vec_bias) in enumerate([
        ("XAU LONG signal",  [1.0, 0.5, 0.2]),
        ("EUR SHORT signal", [0.1, 0.9, 0.1]),
        ("BTC breakout",     [0.5, 0.3, 0.9]),
    ]):
        rec = MemoryRecord.create(label, "trading")
        bias = np.array(vec_bias, dtype=np.float32)
        noise = rng.random(3).astype(np.float32) * 0.1
        vec = (bias + noise)
        vec = vec / np.linalg.norm(vec)
        rec.embedding = vec.tolist()
        rec.quality = "HIGH"
        records.append(rec)

    # Fake retriever (no DB needed for math test)
    class FakeStore:
        def get_with_embeddings(self, **_): return records

    class FakeEmbed:
        def embed(self, text):
            # Query similar to XAU LONG
            vec = np.array([0.9, 0.4, 0.2], dtype=np.float32)
            return (vec / np.linalg.norm(vec)).tolist()

    retriever = MemoryRetriever(
        store=FakeStore(),
        embedding_client=FakeEmbed(),
        min_similarity=0.5,
    )

    results = retriever.recall("XAU gold long signal mean reversion")
    print(f"Query: 'XAU gold long signal mean reversion'")
    print(f"Results ({len(results)}):")
    for r in results:
        print(f"  sim={r.similarity:.4f}  '{r.record.content}'  [{r.relevance_note}]")

    assert len(results) > 0, "Expected at least 1 result"
    assert results[0].record.content == "XAU LONG signal", \
        f"Expected 'XAU LONG signal' as top result, got '{results[0].record.content}'"

    print("\nretriever.py — OK")
