"""
SMA — Context Manager
=====================
Selects and formats the best memories to inject into an LLM prompt,
respecting token budget and quality filters.

Design goals:
  - Never exceed token_budget (prevents context overflow)
  - Prioritize HIGH/MEDIUM over LOW/LUCKY/FALSE_EDGE
    (inject lessons, not patterns to avoid — those go in warnings)
  - Recent memories slightly outweigh older ones (recency bias)
  - Disputed memories are surfaced as warnings, not as facts
  - Output is plain text — agent decides how to embed in its prompt
"""

from dataclasses import dataclass
from typing import List, Optional

from sma.memory_store import MemoryRecord
from sma.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Token estimation (no tiktoken dependency — rough but sufficient)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/mixed text."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Context result
# ---------------------------------------------------------------------------

@dataclass
class ContextBlock:
    """Formatted context ready to be injected into a prompt."""
    memories_text: str          # positive memories to learn from
    warnings_text: str          # LUCKY / FALSE_EDGE / disputed — cautionary
    total_tokens: int
    memory_count: int
    warning_count: int
    sources: List[str]          # memory IDs used


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Builds a structured context block from retrieved memories.

    Usage:
        ctx = ContextManager(token_budget=2000)
        block = ctx.build(retrieval_results, recent_memories)
        # inject block.memories_text into your LLM prompt
    """

    # Quality levels that are safe to present as "learned patterns"
    TRUSTED_LEVELS  = {"HIGH", "MEDIUM"}
    # Levels that should be presented as cautionary warnings
    CAUTION_LEVELS  = {"LOW", "LUCKY", "FALSE_EDGE"}

    def __init__(
        self,
        token_budget: int = 2000,
        recency_weight: float = 0.2,    # boost for recent memories (0 = no boost)
        include_regret_score: bool = True,
        include_similarity: bool = False,
    ):
        self.token_budget = token_budget
        self.recency_weight = recency_weight
        self.include_regret_score = include_regret_score
        self.include_similarity = include_similarity

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        retrieval_results: List[RetrievalResult],
        recent_memories: Optional[List[MemoryRecord]] = None,
    ) -> ContextBlock:
        """
        Build a context block from retrieval results and optional recents.

        Args:
            retrieval_results: From MemoryRetriever.recall() or .surface()
            recent_memories:   Latest memories from current session (no embedding needed)
        """
        # Score and rank all candidates
        ranked = self._rank(retrieval_results, recent_memories or [])

        # Split into trusted and caution buckets
        trusted = [(r, s) for r, s in ranked if r.quality in self.TRUSTED_LEVELS]
        caution  = [(r, s) for r, s in ranked if r.quality in self.CAUTION_LEVELS]
        disputed = [(r, s) for r, s in ranked if getattr(r, "audit_disputed", False)]

        # Build memory text within budget
        budget_left = self.token_budget
        memory_lines: List[str] = []
        warning_lines: List[str] = []
        sources: List[str] = []

        # Trusted memories (main context)
        for record, score in trusted:
            line = self._format_memory(record, score, is_warning=False)
            tokens = _estimate_tokens(line)
            if tokens > budget_left:
                break
            memory_lines.append(line)
            sources.append(record.id)
            budget_left -= tokens

        # Caution memories (smaller budget)
        caution_budget = min(budget_left, self.token_budget // 4)
        for record, score in caution + disputed:
            line = self._format_memory(record, score, is_warning=True)
            tokens = _estimate_tokens(line)
            if tokens > caution_budget:
                break
            warning_lines.append(line)
            sources.append(record.id)
            caution_budget -= tokens

        memories_text = self._format_section(
            "RELEVANT PAST EXPERIENCE", memory_lines
        ) if memory_lines else ""

        warnings_text = self._format_section(
            "CAUTIONARY PATTERNS (do not repeat)", warning_lines
        ) if warning_lines else ""

        total_used = _estimate_tokens(memories_text + warnings_text)

        return ContextBlock(
            memories_text=memories_text,
            warnings_text=warnings_text,
            total_tokens=total_used,
            memory_count=len(memory_lines),
            warning_count=len(warning_lines),
            sources=sources,
        )

    def build_proactive(
        self,
        proactive_results: List[RetrievalResult],
    ) -> str:
        """
        Compact proactive summary — surfaced when a new entry is added.
        Shorter and more direct than full context injection.
        """
        if not proactive_results:
            return ""

        lines = []
        budget = min(self.token_budget // 2, 800)  # proactive is compact

        for result in proactive_results[:3]:
            record = result.record
            quality_label = f"[{record.quality}]"
            sim_label = f" (sim={result.similarity:.2f})" if self.include_similarity else ""
            line = f"• {quality_label}{sim_label} {record.content}"
            if record.outcome:
                outcome_str = self._format_outcome(record.outcome)
                if outcome_str:
                    line += f" → {outcome_str}"
            tokens = _estimate_tokens(line)
            if tokens > budget:
                break
            lines.append(line)
            budget -= tokens

        if not lines:
            return ""

        return "RELATED PAST CONTEXT:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank(
        self,
        retrieval_results: List[RetrievalResult],
        recent_memories: List[MemoryRecord],
    ) -> List[tuple]:
        """Score all memories and return (MemoryRecord, score) sorted by score desc."""
        scored = []

        # Retrieval results — already ranked by similarity
        for i, result in enumerate(retrieval_results):
            base_score = result.similarity
            recency_boost = self.recency_weight * max(0, 1 - i / len(retrieval_results))
            quality_mult = {"HIGH": 1.0, "MEDIUM": 0.8, "LOW": 0.5,
                            "LUCKY": 0.4, "FALSE_EDGE": 0.3}.get(result.record.quality, 0.6)
            scored.append((result.record, base_score * quality_mult + recency_boost))

        # Recent memories — lower base score but high recency
        for i, record in enumerate(recent_memories[:5]):
            if not any(r.id == record.id for r, _ in scored):
                recency_boost = self.recency_weight * (1.0 - i * 0.15)
                quality_mult = {"HIGH": 1.0, "MEDIUM": 0.8}.get(record.quality, 0.5)
                scored.append((record, 0.5 * quality_mult + recency_boost))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _format_memory(
        self,
        record: MemoryRecord,
        score: float,
        is_warning: bool,
    ) -> str:
        quality_label = f"[{record.quality}]"
        regret_label = ""
        if self.include_regret_score and record.regret_score >= 0:
            regret_label = f" regret={record.regret_score:.2f}"

        prefix = "⚠️ " if is_warning else "✓ "
        line = f"{prefix}{quality_label}{regret_label} {record.content}"

        if record.outcome:
            outcome_str = self._format_outcome(record.outcome)
            if outcome_str:
                line += f"\n    Outcome: {outcome_str}"

        if is_warning and record.quality == "FALSE_EDGE":
            line += "\n    [HIGH CONFIDENCE, WRONG OUTCOME — do not repeat this reasoning]"
        elif is_warning and record.quality == "LUCKY":
            line += "\n    [LUCKY OUTCOME — reasoning was flawed, outcome was coincidental]"

        return line

    @staticmethod
    def _format_outcome(outcome: dict) -> str:
        if not outcome:
            return ""
        parts = []
        if "pnl_pct" in outcome:
            pnl = outcome["pnl_pct"]
            parts.append(f"P&L={pnl:+.2f}%")
        if "correct" in outcome:
            parts.append("correct" if outcome["correct"] else "incorrect")
        if "success" in outcome:
            parts.append("success" if outcome["success"] else "failed")
        for k, v in outcome.items():
            if k not in ("pnl_pct", "correct", "success"):
                parts.append(f"{k}={v}")
        return ", ".join(parts[:4])   # cap at 4 fields

    @staticmethod
    def _format_section(header: str, lines: List[str]) -> str:
        if not lines:
            return ""
        body = "\n".join(lines)
        return f"--- {header} ---\n{body}\n---\n"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from sma.memory_store import MemoryRecord
    from sma.retriever import RetrievalResult

    print("=== Context Manager self-test ===\n")

    def make_result(content, quality, similarity, pnl=None, regret=0.1):
        rec = MemoryRecord.create(content, "trading")
        rec.quality = quality
        rec.regret_score = regret
        rec.embedding = [0.1] * 10
        if pnl is not None:
            rec.outcome = {"pnl_pct": pnl}
        return RetrievalResult(record=rec, similarity=similarity,
                               relevance_note=f"similar:{quality}")

    results = [
        make_result("LONG XAU at 3328, OU deviation -2.3σ", "HIGH",     0.92, pnl=3.2),
        make_result("SHORT EUR/USD near resistance",          "MEDIUM",   0.81, pnl=1.1),
        make_result("LONG XAU news blackout ignored",         "FALSE_EDGE", 0.78, pnl=-2.5, regret=0.9),
        make_result("SHORT BTC momentum signal weak",         "LUCKY",    0.72, pnl=2.0, regret=0.6),
        make_result("SHORT GBP weak signal",                  "LOW",      0.68, pnl=-1.8, regret=0.7),
    ]

    ctx_mgr = ContextManager(
        token_budget=1500,
        include_regret_score=True,
        include_similarity=True,
    )
    block = ctx_mgr.build(results)

    print(f"Memory count: {block.memory_count}")
    print(f"Warning count: {block.warning_count}")
    print(f"Total tokens: {block.total_tokens}")
    print(f"Sources: {block.sources}\n")
    print("MEMORIES:")
    print(block.memories_text)
    print("WARNINGS:")
    print(block.warnings_text)

    # Proactive test
    proactive_text = ctx_mgr.build_proactive(results[:2])
    print("PROACTIVE:")
    print(proactive_text)

    print("context_manager.py — OK")
