# Sigma Memory Agent (SMA)

**Persistent, queryable, quality-aware memory middleware for AI agents.**

Built for **Qwen Cloud Hackathon — Track 1: MemoryAgent**

---

## What is SMA?

Most AI agents forget everything between sessions. SMA gives any agent
persistent memory that doesn't just store and retrieve — it **classifies
decision quality** and **learns what patterns to avoid**.

```
Without SMA: Agent repeats the same mistake across sessions
With SMA:    Agent recalls a [FALSE_EDGE] pattern from 3 sessions ago
             and avoids it — without retraining the model
```

## Key Features

| Feature | Description |
|---|---|
| **Persistent memory** | SQLite-backed, survives session restarts |
| **Semantic recall** | Qwen `text-embedding-v4` semantic search |
| **Quality classification** | `HIGH / MEDIUM / LOW / LUCKY / FALSE_EDGE` via Qwen |
| **Dual-model audit** | DeepSeek independently audits regret scores |
| **Proactive surfacing** | Surfaces related memories *before* you ask |
| **Context injection** | One-line LLM prompt enrichment |
| **Domain agnostic** | Trading, legal, creative, research — any domain |
| **Injectable config** | Plug in your own regret function and quality rubric |

## Quick Start

```bash
pip install sigma-memory-agent
# or: pip install -r requirements.txt && pip install -e .
```

```python
from sma import SMAClient

sma = SMAClient(qwen_api_key="sk-...")

# Store a decision
memory_id = sma.remember({
    "content": "LONG XAUUSD at 3328.50 — OU deviation -2.3σ",
    "domain": "trading",
    "confidence": 0.72,
})

# Later: attach the outcome → triggers quality classification
sma.update_outcome(memory_id, {"pnl_pct": 3.2, "correct": True})

# Inject relevant context into your next LLM call
context = sma.context_for("XAU long setup, strong momentum")
prompt = f"{your_system_prompt}\n\n{context}\n\nUser: {user_input}"
```

## Domain Configuration

SMA is domain-agnostic. Inject your domain logic:

```python
# Trading domain
def trading_regret(entry: dict) -> float:
    pnl = entry.get("outcome", {}).get("pnl_pct", 0)
    return max(0.0, min(1.0, -pnl / 5.0))

sma = SMAClient(
    qwen_api_key="sk-...",
    deepseek_api_key="sk-...",   # optional: enables audit
    regret_fn=trading_regret,
    quality_rubric={
        "HIGH":       lambda e: e.get("outcome", {}).get("pnl_pct", 0) > 2.0,
        "FALSE_EDGE": lambda e: (e.get("outcome", {}).get("pnl_pct", 0) < 0
                                 and e.get("confidence", 0) > 0.7),
    },
)
```

## Quality Classification

SMA classifies every decision with an outcome:

| Level | Meaning | Learn from? |
|---|---|---|
| `HIGH` | Clear signal, correct reasoning, good outcome | ✅ Yes |
| `MEDIUM` | Reasonable decision, acceptable outcome | ✅ Cautiously |
| `LOW` | Weak signal, bad outcome | ⚠️ Avoid |
| `LUCKY` | Good outcome, **wrong reasoning** | ❌ Dangerous |
| `FALSE_EDGE` | High confidence, **wrong outcome** | ❌ Most dangerous |

## Dual-Model Audit (Qwen + DeepSeek)

Single-model classification risks systematic bias — especially confusing
`LUCKY` with `HIGH`. SMA uses DeepSeek as an independent auditor:

```
Qwen classifies → DeepSeek audits → Consensus or DISPUTED flag
```

- **AGREE (sim ≤ 1 level)**: high confidence result (0.95)
- **DISAGREE (major)**: entry flagged as `audit_disputed=True` → human review queue
- **SKIPPED** (no DeepSeek key): Qwen result used, confidence 0.7

## API Reference

```python
# Store
memory_id = sma.remember(entry, session_id="session_1")
quality   = sma.update_outcome(memory_id, outcome)

# Retrieve
results   = sma.recall("query", domain="trading", min_quality="MEDIUM")
context   = sma.context_for("situation", max_tokens=1000)
related   = sma.surface_related(new_entry)       # proactive

# Inspect
stats     = sma.stats(domain="trading")
disputed  = sma.get_disputed()
high_regret = sma.get_high_regret(min_regret=0.7)

# Maintenance
pending   = sma.pending_classification()
classified = sma.run_pending_classification()
indexed   = sma.reindex()
```

## Run the Demo

```bash
# No API key needed:
python examples/paper_trading_demo.py --offline

# With Qwen API (uses ~5K tokens ≈ $0.002):
python examples/paper_trading_demo.py

# Basic usage:
python examples/basic_usage.py --offline
```

## Run Tests

```bash
python tests/test_client.py
# or:
pytest tests/ -v
```

## Architecture

```
SMAClient (public API)
├── MemoryStore      → SQLite persistence, CRUD, quality-scoped queries
├── MemoryClassifier → Qwen primary + DeepSeek audit + consensus resolution
├── MemoryRetriever  → text-embedding-v4 semantic search (numpy cosine)
└── ContextManager   → token-budget-aware context injection
```

## Models Used

| Model | Purpose |
|---|---|
| `qwen3.7-plus` | Quality classification |
| `text-embedding-v4` | Semantic memory retrieval |
| `deepseek-chat` | Independent regret score audit (optional) |

All via OpenAI-compatible endpoints — no extra SDK needed.

## Governed Extensions (Not Included by Design)

The following are intentionally excluded pending proper governance layers:

- **Meaning Extractor**: requires ≥500 HIGH-quality entries + human review
- **Counterfactual Engine**: requires verifiable outcome tracking
- **Strategy Evolver**: requires human-in-the-loop + cross-model bias audit

Activating these without governance risks generating **confident but
incorrect learned patterns** — more dangerous than no learning at all.

## License

MIT — build whatever you want on top.
