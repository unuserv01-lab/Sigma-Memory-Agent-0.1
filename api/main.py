"""
SMA — FastAPI REST API
=======================
Production-ready HTTP wrapper around SMAClient.
Designed for deployment on Alibaba Cloud Function Compute (FC).

ENDPOINTS:
  POST /memory              → remember() — store a new memory
  POST /memory/{id}/outcome → update_outcome() — attach result + classify
  GET  /memory/{id}         → get() — retrieve single memory
  POST /recall              → recall() — semantic search
  POST /context             → context_for() — LLM-ready context string
  POST /surface             → surface_related() — proactive surfacing
  GET  /stats               → stats() — aggregate metrics
  GET  /disputed            → get_disputed() — needs human review
  GET  /high-regret         → get_high_regret() — worst decisions
  GET  /health              → health check (proves Alibaba Cloud deployment)

DEPLOYMENT:
  Local:   uvicorn api.main:app --host 0.0.0.0 --port 8000
  FC:      entry_point = api.main.handler (via Mangum or FC adapter)
  Docker:  see Dockerfile in project root
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sma import SMAClient, SMAConfig


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class RememberRequest(BaseModel):
    content: str
    domain: str = "general"
    session_id: str = "default"
    outcome: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"json_schema_extra": {"example": {
        "content": "LONG XAUUSD at 3328.50 — OU deviation -2.3σ, H4 bullish engulfing",
        "domain": "trading",
        "session_id": "session_001",
        "confidence": 0.72,
        "metadata": {"symbol": "XAUUSD", "action": "ENTER_LONG"},
    }}}


class UpdateOutcomeRequest(BaseModel):
    outcome: Dict[str, Any]
    auto_classify: bool = True

    model_config = {"json_schema_extra": {"example": {
        "outcome": {"pnl_pct": 3.2, "correct": True, "bars_held": 6},
        "auto_classify": True,
    }}}


class RecallRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    min_quality: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=20)
    as_context: bool = False

    model_config = {"json_schema_extra": {"example": {
        "query": "XAU long setup with mean reversion signal",
        "domain": "trading",
        "min_quality": "MEDIUM",
        "limit": 5,
    }}}


class ContextRequest(BaseModel):
    situation: str
    domain: Optional[str] = None
    max_tokens: int = Field(default=2000, ge=100, le=8000)

    model_config = {"json_schema_extra": {"example": {
        "situation": "About to enter SHORT EURUSD, CPI in 12 minutes",
        "domain": "trading",
        "max_tokens": 1500,
    }}}


class SurfaceRequest(BaseModel):
    entry: Dict[str, Any]
    limit: int = Field(default=3, ge=1, le=10)

    model_config = {"json_schema_extra": {"example": {
        "entry": {
            "content": "LONG XAUUSD setup forming",
            "domain": "trading",
            "confidence": 0.68,
        },
        "limit": 3,
    }}}


class RememberResponse(BaseModel):
    memory_id: str
    status: str
    domain: str
    classified: bool


class MemoryResponse(BaseModel):
    id: str
    content: str
    domain: str
    session_id: str
    quality: str
    regret_score: float
    outcome: Optional[Dict[str, Any]]
    audit_consensus: bool
    audit_disputed: bool
    timestamp: str


class RecallItem(BaseModel):
    memory_id: str
    content: str
    domain: str
    quality: str
    regret_score: float
    similarity: float
    relevance_note: str
    outcome: Optional[Dict[str, Any]]


class QualityResultResponse(BaseModel):
    classification: str
    regret_score: float
    qwen_classification: str
    deepseek_verdict: Optional[str]
    consensus: bool
    disputed: bool
    confidence: float
    reasoning: str


class StatsResponse(BaseModel):
    total: int
    by_quality: Dict[str, int]
    by_domain: Dict[str, int]
    avg_regret_score: float
    disputed_count: int
    unclassified_count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    alibaba_cloud: bool
    qwen_configured: bool
    deepseek_audit: bool
    total_memories: int
    region: str


# ---------------------------------------------------------------------------
# App factory + lifespan
# ---------------------------------------------------------------------------

_sma_client: Optional[SMAClient] = None


def get_sma() -> SMAClient:
    if _sma_client is None:
        raise HTTPException(status_code=503, detail="SMA client not initialized")
    return _sma_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SMAClient on startup."""
    global _sma_client

    qwen_key     = os.getenv("DASHSCOPE_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    db_path      = os.getenv("SMA_DB_PATH", "sma_memory.db")

    _sma_client = SMAClient(
        qwen_api_key=qwen_key or None,
        deepseek_api_key=deepseek_key or None,
        db_path=db_path,
    )

    print(f"[SMA] Initialized | db={db_path} | "
          f"qwen={'yes' if qwen_key else 'no'} | "
          f"deepseek={'yes' if deepseek_key else 'no'}")
    yield

    # Cleanup on shutdown (SQLite closes automatically)
    print("[SMA] Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sigma Memory Agent (SMA)",
        description="""
Persistent, queryable, quality-aware memory middleware for AI agents.

Built for **Qwen Cloud Hackathon — Track 1: MemoryAgent**.
Deployed on **Alibaba Cloud Function Compute**.

Key features:
- Semantic memory storage via `text-embedding-v4`
- Quality classification (HIGH/MEDIUM/LUCKY/FALSE_EDGE) via `qwen3.7-plus`
- Dual-model regret score audit (Qwen + DeepSeek)
- Cross-session learning without model retraining
- Domain-agnostic (trading, legal, creative, research)
        """,
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint.
    Proves deployment on Alibaba Cloud and shows configuration status.
    """
    sma = get_sma()
    stats = sma.stats()

    return HealthResponse(
        status="healthy",
        service="sigma-memory-agent",
        version="1.0.0",
        alibaba_cloud=True,
        qwen_configured=sma._cfg.qwen_api_key != "",
        deepseek_audit=sma._cfg.deepseek_api_key != "",
        total_memories=stats.total,
        region=os.getenv("ALIBABA_CLOUD_REGION", "ap-southeast-1"),
    )


@app.post("/memory", response_model=RememberResponse, tags=["Memory"])
async def remember(request: RememberRequest):
    """
    Store a new memory.

    If `outcome` is provided, classification runs immediately.
    Otherwise, classification runs when you call `PUT /memory/{id}/outcome`.
    """
    sma = get_sma()

    entry = {
        "content":    request.content,
        "domain":     request.domain,
        "session_id": request.session_id,
    }
    if request.outcome:
        entry["outcome"] = request.outcome
    if request.confidence is not None:
        entry["confidence"] = request.confidence
    if request.metadata:
        entry.update(request.metadata)

    memory_id = sma.remember(entry, session_id=request.session_id)

    record = sma._store.get(memory_id)
    classified = record.quality != "UNCLASSIFIED" if record else False

    return RememberResponse(
        memory_id=memory_id,
        status="stored",
        domain=request.domain,
        classified=classified,
    )


@app.post(
    "/memory/{memory_id}/outcome",
    response_model=Optional[QualityResultResponse],
    tags=["Memory"],
)
async def update_outcome(memory_id: str, request: UpdateOutcomeRequest):
    """
    Attach an outcome to an existing memory and trigger quality classification.

    Call this when a decision's result is known (e.g. trade closed, prediction resolved).
    Returns the quality classification result, or null if classifier not configured.
    """
    sma = get_sma()

    record = sma._store.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    result = sma.update_outcome(
        memory_id,
        request.outcome,
        auto_classify=request.auto_classify,
    )

    if result is None:
        return None

    return QualityResultResponse(
        classification=result.classification,
        regret_score=result.regret_score,
        qwen_classification=result.qwen_classification,
        deepseek_verdict=result.deepseek_verdict,
        consensus=result.consensus,
        disputed=result.disputed,
        confidence=result.confidence,
        reasoning=result.qwen_reasoning,
    )


@app.get("/memory/{memory_id}", response_model=MemoryResponse, tags=["Memory"])
async def get_memory(memory_id: str):
    """Retrieve a single memory by ID."""
    sma = get_sma()
    record = sma._store.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    return MemoryResponse(
        id=record.id,
        content=record.content,
        domain=record.domain,
        session_id=record.session_id,
        quality=record.quality,
        regret_score=record.regret_score,
        outcome=record.outcome,
        audit_consensus=record.audit_consensus,
        audit_disputed=record.audit_disputed,
        timestamp=record.timestamp,
    )


@app.post("/recall", response_model=List[RecallItem], tags=["Retrieval"])
async def recall(request: RecallRequest):
    """
    Semantic search over stored memories.

    Returns memories ordered by relevance to the query.
    Filter by domain and minimum quality level.
    """
    sma = get_sma()

    results = sma.recall(
        query=request.query,
        domain=request.domain,
        min_quality=request.min_quality,
        limit=request.limit,
        as_context=False,
    )

    if not results:
        return []

    items = []
    for r in results:
        items.append(RecallItem(
            memory_id=r.record.id,
            content=r.record.content,
            domain=r.record.domain,
            quality=r.record.quality,
            regret_score=r.record.regret_score,
            similarity=r.similarity,
            relevance_note=r.relevance_note,
            outcome=r.record.outcome,
        ))
    return items


@app.post("/context", tags=["Retrieval"])
async def context_for(request: ContextRequest):
    """
    Get LLM-ready context string for injection into a prompt.

    Returns formatted text with relevant memories + cautionary warnings.
    Respects token budget. Paste directly into your system prompt.
    """
    sma = get_sma()
    context = sma.context_for(
        situation=request.situation,
        domain=request.domain,
        max_tokens=request.max_tokens,
    )
    return {"context": context, "length": len(context)}


@app.post("/surface", tags=["Retrieval"])
async def surface_related(request: SurfaceRequest):
    """
    Proactively surface memories related to a new entry BEFORE storing it.

    Call this before `POST /memory` to give your agent relevant past context first.
    This is SMA's proactive recall feature — agent automatically gets reminded
    of relevant past decisions without explicitly querying.
    """
    sma = get_sma()
    text = sma.surface_related(request.entry, limit=request.limit)
    return {"proactive_context": text, "has_related": bool(text)}


@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
async def stats(domain: Optional[str] = Query(default=None)):
    """Aggregate statistics for the memory store, optionally filtered by domain."""
    sma = get_sma()
    s = sma.stats(domain=domain)
    return StatsResponse(
        total=s.total,
        by_quality=s.by_quality,
        by_domain=s.by_domain,
        avg_regret_score=s.avg_regret_score,
        disputed_count=s.disputed_count,
        unclassified_count=s.unclassified_count,
    )


@app.get("/disputed", response_model=List[MemoryResponse], tags=["Analytics"])
async def get_disputed(limit: int = Query(default=20, ge=1, le=100)):
    """
    Memories where Qwen and DeepSeek disagreed on quality classification.
    These are flagged for human review.
    """
    sma = get_sma()
    records = sma.get_disputed(limit=limit)
    return [
        MemoryResponse(
            id=r.id, content=r.content, domain=r.domain,
            session_id=r.session_id, quality=r.quality,
            regret_score=r.regret_score, outcome=r.outcome,
            audit_consensus=r.audit_consensus,
            audit_disputed=r.audit_disputed, timestamp=r.timestamp,
        )
        for r in records
    ]


@app.get("/high-regret", response_model=List[MemoryResponse], tags=["Analytics"])
async def get_high_regret(
    domain: Optional[str] = Query(default=None),
    min_regret: float = Query(default=0.7, ge=0.0, le=1.0),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Highest regret decisions — patterns to learn from and avoid.
    """
    sma = get_sma()
    records = sma.get_high_regret(
        domain=domain, min_regret=min_regret, limit=limit
    )
    return [
        MemoryResponse(
            id=r.id, content=r.content, domain=r.domain,
            session_id=r.session_id, quality=r.quality,
            regret_score=r.regret_score, outcome=r.outcome,
            audit_consensus=r.audit_consensus,
            audit_disputed=r.audit_disputed, timestamp=r.timestamp,
        )
        for r in records
    ]


@app.post("/reindex", tags=["Maintenance"])
async def reindex(limit: int = Query(default=100, ge=1, le=500)):
    """
    Batch-embed unindexed memories to make them searchable.
    Run periodically to keep the semantic search index fresh.
    """
    sma = get_sma()
    count = sma.reindex(limit=limit)
    return {"indexed": count, "message": f"Embedded {count} memories"}


@app.post("/classify-pending", tags=["Maintenance"])
async def classify_pending(limit: int = Query(default=50, ge=1, le=200)):
    """
    Classify all memories that have outcomes but haven't been classified yet.
    """
    sma = get_sma()
    count = sma.run_pending_classification(limit=limit)
    pending = sma.pending_classification()
    return {
        "classified": count,
        "still_pending": pending,
        "message": f"Classified {count} memories",
    }


# ---------------------------------------------------------------------------
# Alibaba Cloud FC handler (for serverless deployment)
# ---------------------------------------------------------------------------
# FC expects a handler function. When deploying to Function Compute,
# set the handler to: api.main.handler

try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="on")
except ImportError:
    # Mangum not installed — direct uvicorn deployment
    handler = None


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
