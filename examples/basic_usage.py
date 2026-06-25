"""
SMA — Basic Usage (Quickstart)
===============================
Minimal example showing SMA in any domain.
No domain-specific config needed — works out of the box.

Usage:
    python examples/basic_usage.py --offline
    python examples/basic_usage.py          # requires DASHSCOPE_API_KEY
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sma import SMAClient, SMAConfig


def run(offline: bool = False):
    print("=== SMA Basic Usage ===\n")

    sma = SMAClient(
        qwen_api_key="" if offline else os.getenv("DASHSCOPE_API_KEY"),
        deepseek_api_key="" if offline else os.getenv("DEEPSEEK_API_KEY"),
        db_path="basic_demo.db",
        config=SMAConfig(embed_on_store=not offline),
    )

    # 1. Store memories
    print("1. Storing memories...")
    id1 = sma.remember({
        "content": "User prefers concise bullet-point summaries over paragraphs",
        "domain": "user_preferences",
    })
    id2 = sma.remember({
        "content": "User asked about Python async patterns — context: FastAPI backend",
        "domain": "user_preferences",
        "outcome": {"helpful": True, "user_rating": 5},
    })
    id3 = sma.remember({
        "content": "Explained REST vs GraphQL — user found REST explanation clearer",
        "domain": "user_preferences",
        "outcome": {"helpful": True, "user_rating": 4},
    })
    print(f"   Stored 3 memories: {id1[:8]}... {id2[:8]}... {id3[:8]}...")

    # 2. Update outcome later
    print("\n2. Updating outcome after result is known...")
    sma.update_outcome(id1, {"applied": True, "improved_engagement": True},
                       auto_classify=not offline)
    print("   Outcome updated for memory 1")

    # 3. Recall semantically similar memories
    print("\n3. Recall: 'what does this user prefer for explanations?'")
    if not offline and sma._retriever:
        results = sma.recall("user preferences for explanations and communication style")
        for r in results:
            print(f"   [{r.record.quality}] sim={r.similarity:.2f} — {r.record.content[:70]}...")
    else:
        records = sma._store.get_by_domain("user_preferences", limit=3)
        print(f"   Found {len(records)} memories (offline: no semantic search)")
        for r in records:
            print(f"   [{r.quality}] {r.content[:70]}...")

    # 4. Get context for LLM injection
    print("\n4. Context injection for next LLM call:")
    ctx = sma.context_for("user asking about database design patterns", max_tokens=500)
    if ctx:
        print(ctx[:400] + "..." if len(ctx) > 400 else ctx)
    else:
        print("   (empty — embed_on_store=False in offline mode)")

    # 5. Stats
    print("\n5. Stats:")
    stats = sma.stats()
    print(f"   Total: {stats.total} | By quality: {stats.by_quality}")
    print(f"   Avg regret: {stats.avg_regret_score:.3f}")

    # Cleanup
    import os as _os
    try:
        _os.remove("basic_demo.db")
    except Exception:
        pass

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run(offline=args.offline)
