"""
SMA — Demo Endpoint
====================
One-click demo endpoint for hackathon judges.
Runs the full cross-session learning scenario without needing
to clone the repo or use a terminal.

Clearly separated from production endpoints — uses an isolated
demo database that resets each time, so judges can run it
repeatedly without polluting real data.
"""

import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter

from sma import SMAClient, SMAConfig


router = APIRouter(prefix="/demo", tags=["Demo (Judges Start Here)"])


# ---------------------------------------------------------------------------
# Demo scenario data — same as examples/paper_trading_demo.py
# ---------------------------------------------------------------------------

DEMO_SIGNALS = [
    {
        "session": "session_1",
        "content": "LONG XAUUSD at 3328.50 — OU deviation -2.3σ, H4 momentum positive",
        "domain": "trading_demo",
        "confidence": 0.72,
        "signal_followed": True,
        "outcome": {"pnl_pct": 3.2, "correct": True},
    },
    {
        "session": "session_1",
        "content": "SHORT EURUSD at 1.0850 — news blackout IGNORED (NFP in 10min)",
        "domain": "trading_demo",
        "confidence": 0.78,
        "signal_followed": False,
        "outcome": {"pnl_pct": -2.5, "correct": False},
    },
    {
        "session": "session_2",
        "content": "LONG XAUUSD at 3315.20 — OU deviation -1.8σ, news calendar clear",
        "domain": "trading_demo",
        "confidence": 0.68,
        "signal_followed": True,
        "outcome": {"pnl_pct": 2.1, "correct": True},
    },
    {
        "session": "session_3",
        "content": "SHORT EURUSD at 1.0780 — CPI release in 8 minutes, signal strong",
        "domain": "trading_demo",
        "confidence": 0.76,
        "signal_followed": False,
        "outcome": {"pnl_pct": -1.9, "correct": False},
    },
]


def trading_regret_fn(entry: dict) -> float:
    outcome = entry.get("outcome", {})
    pnl = float(outcome.get("pnl_pct", 0))
    confidence = float(entry.get("confidence", 0.5))
    if pnl > 2.0:
        return 0.0
    if pnl > 0:
        return max(0.0, 0.3 - pnl * 0.1)
    if pnl >= -1.0:
        return 0.4 + confidence * 0.3
    return min(1.0, 0.6 + abs(pnl) * 0.08)


TRADING_QUALITY_RUBRIC = {
    "HIGH": lambda e: (
        e.get("outcome", {}).get("pnl_pct", 0) > 2.0
        and e.get("signal_followed", True)
    ),
    "MEDIUM": lambda e: (
        0 < e.get("outcome", {}).get("pnl_pct", 0) <= 2.0
        and e.get("signal_followed", True)
    ),
    "LUCKY": lambda e: (
        e.get("outcome", {}).get("pnl_pct", 0) > 0
        and not e.get("signal_followed", True)
    ),
    "FALSE_EDGE": lambda e: (
        e.get("outcome", {}).get("pnl_pct", 0) < 0
        and e.get("confidence", 0) > 0.7
    ),
}


# ---------------------------------------------------------------------------
# Demo endpoint
# ---------------------------------------------------------------------------

@router.post("/run-scenario")
async def run_demo_scenario():
    """
    🎯 START HERE — One-click demo for judges.

    Runs a complete 3-session cross-session learning scenario:
    - Session 1: Agent makes a FALSE_EDGE mistake (ignores news blackout)
    - Session 2: Agent makes a clean HIGH quality decision
    - Session 3: Agent faces the SAME situation as Session 1's mistake

    Shows whether SMA successfully recalls the Session 1 FALSE_EDGE
    and surfaces it as a warning before Session 3's decision.

    Uses an isolated, temporary database — safe to run multiple times.
    Does NOT affect production memory data.
    """
    demo_db = os.path.join(tempfile.gettempdir(), f"sma_demo_{datetime.now().timestamp()}.db")

    qwen_key = os.getenv("DASHSCOPE_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")

    demo_sma = SMAClient(
        qwen_api_key=qwen_key or None,
        deepseek_api_key=deepseek_key or None,
        db_path=demo_db,
        regret_fn=trading_regret_fn,
        quality_rubric=TRADING_QUALITY_RUBRIC,
    )

    timeline: List[Dict] = []

    # Phase 1: Store session 1 and 2 decisions
    for sig in DEMO_SIGNALS[:3]:
        session_id = sig.pop("session")
        outcome = sig.pop("outcome")

        memory_id = demo_sma.remember(
            {**sig, "outcome": outcome},
            session_id=session_id,
            auto_classify=bool(qwen_key),
        )
        record = demo_sma._store.get(memory_id)

        timeline.append({
            "session": session_id,
            "action": "stored_decision",
            "content": sig["content"],
            "outcome_pnl": outcome["pnl_pct"],
            "quality": record.quality if record else "UNCLASSIFIED",
            "regret_score": record.regret_score if record else None,
        })

    # Phase 2: Session 3 — same mistake pattern as Session 1
    session_3_situation = (
        "About to SHORT EURUSD, high confidence signal, "
        "CPI release in 12 minutes — same setup as a past loss"
    )

    proactive_context = ""
    if qwen_key:
        proactive_context = demo_sma.surface_related({
            "content": session_3_situation,
            "domain": "trading_demo",
        })

    context_for_llm = ""
    if qwen_key:
        context_for_llm = demo_sma.context_for(
            session_3_situation, domain="trading_demo", max_tokens=800
        )

    timeline.append({
        "session": "session_3",
        "action": "cross_session_recall",
        "situation": session_3_situation,
        "proactive_context_surfaced": proactive_context or "(requires DASHSCOPE_API_KEY for live recall)",
        "context_injected_to_llm": context_for_llm or "(requires DASHSCOPE_API_KEY for live recall)",
    })

    # Phase 3: Stats
    stats = demo_sma.stats(domain="trading_demo")

    # Cleanup
    try:
        os.remove(demo_db)
    except Exception:
        pass

    return {
        "demo_complete": True,
        "explanation": (
            "Without SMA: An agent would repeat the Session 1 mistake in "
            "Session 3 (same FALSE_EDGE pattern: high confidence + news "
            "blackout ignored). With SMA: the FALSE_EDGE memory from "
            "Session 1 is recalled and injected as a warning BEFORE the "
            "Session 3 decision is made — enabling cross-session learning "
            "without retraining the model."
        ),
        "timeline": timeline,
        "memory_stats": {
            "total_memories": stats.total,
            "by_quality": stats.by_quality,
            "avg_regret_score": stats.avg_regret_score,
        },
        "models_used": {
            "classification": "qwen3.7-plus" if qwen_key else "not configured",
            "embeddings": "text-embedding-v4" if qwen_key else "not configured",
            "audit": "deepseek-chat" if deepseek_key else "not configured (optional)",
        },
        "note": "This demo used an isolated temporary database. "
                "No production data was affected.",
    }


@router.get("/explain")
async def explain_demo():
    """
    📖 What does /demo/run-scenario actually test?

    Read this first if you want to understand the scenario
    before running it.
    """
    return {
        "track": "Track 1: MemoryAgent — Qwen Cloud Hackathon",
        "what_this_demonstrates": [
            "Persistent memory across multiple sessions (not just one conversation)",
            "Quality classification: HIGH / MEDIUM / LUCKY / FALSE_EDGE",
            "Proactive memory surfacing (agent gets reminded without asking)",
            "Context injection ready for any LLM prompt",
            "Dual-model regret score audit (Qwen primary + DeepSeek auditor)",
        ],
        "scenario": (
            "A trading agent makes a risky decision in Session 1 that goes "
            "wrong (classified FALSE_EDGE — high confidence, bad outcome, "
            "ignored a known risk factor). Two sessions later, the agent "
            "faces a nearly identical situation. SMA recalls the past "
            "FALSE_EDGE and surfaces it as a warning before the new "
            "decision — this is cross-session learning without "
            "fine-tuning or retraining any model."
        ),
        "try_it": "POST /demo/run-scenario",
    }
