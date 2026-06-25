"""
SMA — Test Suite (Offline)
===========================
All tests run without API calls.
Tests storage, classification logic, context management, and client interface.

Usage:
    python -m pytest tests/ -v
    python tests/test_client.py   # run directly
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sma.memory_store import MemoryRecord, MemoryStore
from sma.classifier import MemoryClassifier, QualityLevel, QualityResult
from sma.context_manager import ContextManager
from sma.retriever import RetrievalResult
from sma.client import SMAClient, SMAConfig


# ---------------------------------------------------------------------------
# MemoryStore tests
# ---------------------------------------------------------------------------

class TestMemoryStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(db_path=os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_insert_and_get(self):
        rec = MemoryRecord.create("Test content", domain="test")
        mid = self.store.insert(rec)
        self.assertIsNotNone(mid)

        retrieved = self.store.get(mid)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Test content")
        self.assertEqual(retrieved.domain, "test")
        self.assertEqual(retrieved.quality, "UNCLASSIFIED")

    def test_update_quality(self):
        rec = MemoryRecord.create("Test", domain="test")
        mid = self.store.insert(rec)

        ok = self.store.update_quality(mid, "HIGH", 0.05,
                                        audit_consensus=True, audit_disputed=False)
        self.assertTrue(ok)

        updated = self.store.get(mid)
        self.assertEqual(updated.quality, "HIGH")
        self.assertAlmostEqual(updated.regret_score, 0.05)
        self.assertTrue(updated.audit_consensus)
        self.assertFalse(updated.audit_disputed)

    def test_update_outcome(self):
        rec = MemoryRecord.create("Test", domain="test")
        mid = self.store.insert(rec)

        outcome = {"pnl_pct": 2.5, "correct": True}
        ok = self.store.update_outcome(mid, outcome)
        self.assertTrue(ok)

        updated = self.store.get(mid)
        self.assertEqual(updated.outcome, outcome)

    def test_get_by_domain(self):
        for i in range(3):
            r = MemoryRecord.create(f"Trading {i}", domain="trading")
            self.store.insert(r)
        for i in range(2):
            r = MemoryRecord.create(f"Legal {i}", domain="legal")
            self.store.insert(r)

        trading = self.store.get_by_domain("trading")
        self.assertEqual(len(trading), 3)
        legal = self.store.get_by_domain("legal")
        self.assertEqual(len(legal), 2)

    def test_get_by_session(self):
        for i in range(4):
            r = MemoryRecord.create(f"Item {i}", domain="test",
                                    session_id="session_A")
            self.store.insert(r)
        r = MemoryRecord.create("Other", domain="test", session_id="session_B")
        self.store.insert(r)

        session_a = self.store.get_by_session("session_A")
        self.assertEqual(len(session_a), 4)
        session_b = self.store.get_by_session("session_B")
        self.assertEqual(len(session_b), 1)

    def test_quality_filter(self):
        records_data = [
            ("HIGH entry",   "HIGH"),
            ("MEDIUM entry", "MEDIUM"),
            ("LOW entry",    "LOW"),
            ("LUCKY entry",  "LUCKY"),
        ]
        for content, quality in records_data:
            r = MemoryRecord.create(content, domain="test")
            mid = self.store.insert(r)
            self.store.update_quality(mid, quality, 0.5)

        high_only = self.store.get_by_domain("test", min_quality="HIGH")
        self.assertEqual(len(high_only), 1)
        self.assertEqual(high_only[0].quality, "HIGH")

        medium_and_above = self.store.get_by_domain("test", min_quality="MEDIUM")
        qualities = {r.quality for r in medium_and_above}
        self.assertIn("HIGH", qualities)
        self.assertIn("MEDIUM", qualities)
        self.assertNotIn("LOW", qualities)
        self.assertNotIn("LUCKY", qualities)

    def test_unclassified_with_outcome(self):
        r1 = MemoryRecord.create("Has outcome", domain="test")
        mid1 = self.store.insert(r1)
        self.store.update_outcome(mid1, {"pnl_pct": 1.0})

        r2 = MemoryRecord.create("No outcome", domain="test")
        self.store.insert(r2)

        pending = self.store.get_unclassified_with_outcome()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].id, mid1)

    def test_get_disputed(self):
        r = MemoryRecord.create("Disputed", domain="test")
        mid = self.store.insert(r)
        self.store.update_quality(mid, "HIGH", 0.1, audit_disputed=True)

        disputed = self.store.get_disputed()
        self.assertEqual(len(disputed), 1)
        self.assertTrue(disputed[0].audit_disputed)

    def test_stats(self):
        for quality in ["HIGH", "HIGH", "MEDIUM", "LOW"]:
            r = MemoryRecord.create("Item", domain="stats_test")
            mid = self.store.insert(r)
            self.store.update_quality(mid, quality, 0.3)

        stats = self.store.stats(domain="stats_test")
        self.assertEqual(stats.total, 4)
        self.assertEqual(stats.by_quality.get("HIGH"), 2)
        self.assertEqual(stats.by_quality.get("MEDIUM"), 1)
        self.assertEqual(stats.by_quality.get("LOW"), 1)

    def test_delete(self):
        r = MemoryRecord.create("To delete", domain="test")
        mid = self.store.insert(r)
        ok = self.store.delete(mid)
        self.assertTrue(ok)
        self.assertIsNone(self.store.get(mid))

    def test_get_high_regret(self):
        for regret in [0.2, 0.6, 0.85, 0.95]:
            r = MemoryRecord.create(f"regret={regret}", domain="test")
            mid = self.store.insert(r)
            self.store.update_quality(mid, "LOW", regret)

        high = self.store.get_high_regret(domain="test", min_regret=0.7)
        self.assertEqual(len(high), 2)
        self.assertGreaterEqual(high[0].regret_score, 0.7)


# ---------------------------------------------------------------------------
# Classifier tests (offline — tests rubric + consensus logic)
# ---------------------------------------------------------------------------

class TestClassifierOffline(unittest.TestCase):

    def _make_classifier(self):
        """Build a classifier with rubric only (no API calls)."""
        c = MemoryClassifier.__new__(MemoryClassifier)
        c._deepseek = None
        c.regret_fn = None
        c.audit_rubric = False
        c.quality_rubric = {
            "HIGH":       lambda e: e.get("outcome", {}).get("pnl_pct", 0) > 2.0,
            "MEDIUM":     lambda e: 0 < e.get("outcome", {}).get("pnl_pct", 0) <= 2.0,
            "LOW":        lambda e: e.get("outcome", {}).get("pnl_pct", 0) < 0,
            "LUCKY":      lambda e: (e.get("outcome", {}).get("pnl_pct", 0) > 0
                                     and not e.get("signal_followed", True)),
            "FALSE_EDGE": lambda e: (e.get("outcome", {}).get("pnl_pct", 0) < 0
                                     and e.get("confidence", 0) > 0.7),
        }
        return c

    def test_rubric_high(self):
        c = self._make_classifier()
        cls, _ = c._rubric_classify({"outcome": {"pnl_pct": 3.0}, "signal_followed": True})
        self.assertEqual(cls, "HIGH")

    def test_rubric_medium(self):
        c = self._make_classifier()
        cls, _ = c._rubric_classify({"outcome": {"pnl_pct": 1.5}, "signal_followed": True})
        self.assertEqual(cls, "MEDIUM")

    def test_rubric_false_edge(self):
        c = self._make_classifier()
        cls, _ = c._rubric_classify({"outcome": {"pnl_pct": -2.0}, "confidence": 0.85})
        self.assertEqual(cls, "FALSE_EDGE")

    def test_rubric_lucky(self):
        c = self._make_classifier()
        cls, _ = c._rubric_classify({"outcome": {"pnl_pct": 1.0}, "signal_followed": False})
        self.assertEqual(cls, "LUCKY")

    def test_regret_compute_high_pnl(self):
        c = self._make_classifier()
        c.regret_fn = None
        regret = c._compute_regret({"outcome": {"pnl_pct": 3.5}}, "HIGH")
        self.assertEqual(regret, 0.0)

    def test_regret_compute_big_loss(self):
        c = self._make_classifier()
        c.regret_fn = None
        regret = c._compute_regret({"outcome": {"pnl_pct": -3.0}}, "FALSE_EDGE")
        self.assertGreater(regret, 0.7)

    def test_regret_fn_override(self):
        c = self._make_classifier()
        c.regret_fn = lambda e: 0.42
        regret = c._compute_regret({"outcome": {"pnl_pct": 5.0}}, "HIGH")
        self.assertAlmostEqual(regret, 0.42)

    def test_consensus_agree(self):
        c = self._make_classifier()
        final, consensus, disputed, conf = c._resolve("HIGH", "AGREE", "HIGH")
        self.assertEqual(final, "HIGH")
        self.assertTrue(consensus)
        self.assertFalse(disputed)
        self.assertAlmostEqual(conf, 0.95)

    def test_consensus_minor_disagree(self):
        c = self._make_classifier()
        final, consensus, disputed, conf = c._resolve("HIGH", "DISAGREE", "MEDIUM")
        self.assertFalse(disputed)
        self.assertAlmostEqual(conf, 0.75)

    def test_consensus_major_disagree(self):
        c = self._make_classifier()
        final, consensus, disputed, conf = c._resolve("HIGH", "DISAGREE", "FALSE_EDGE")
        self.assertTrue(disputed)
        self.assertAlmostEqual(conf, 0.4)

    def test_consensus_skipped(self):
        c = self._make_classifier()
        final, consensus, disputed, conf = c._resolve("HIGH", "SKIPPED", None)
        self.assertEqual(final, "HIGH")
        self.assertFalse(consensus)
        self.assertFalse(disputed)

    def test_unclassified_no_outcome(self):
        c = self._make_classifier()
        result = c.classify({"content": "some decision"})  # no outcome
        self.assertEqual(result.classification, "UNCLASSIFIED")
        self.assertEqual(result.regret_score, -1.0)

    def test_parse_classification_response(self):
        valid_json = '{"classification": "HIGH", "reasoning": "clear signal"}'
        cls, reason = MemoryClassifier._parse_classification_response(valid_json)
        self.assertEqual(cls, "HIGH")
        self.assertEqual(reason, "clear signal")

    def test_parse_classification_invalid(self):
        cls, reason = MemoryClassifier._parse_classification_response("not json at all")
        self.assertEqual(cls, "MEDIUM")  # safe default

    def test_parse_audit_response(self):
        valid = '{"verdict": "AGREE", "classification": "HIGH", "reasoning": "ok"}'
        v, c, r = MemoryClassifier._parse_audit_response(valid)
        self.assertEqual(v, "AGREE")
        self.assertEqual(c, "HIGH")


# ---------------------------------------------------------------------------
# ContextManager tests
# ---------------------------------------------------------------------------

class TestContextManager(unittest.TestCase):

    def _make_result(self, content, quality, similarity, pnl=None):
        rec = MemoryRecord.create(content, "trading")
        rec.quality = quality
        rec.regret_score = 0.1 if quality == "HIGH" else 0.7
        rec.embedding = [0.1] * 8
        if pnl is not None:
            rec.outcome = {"pnl_pct": pnl}
        return RetrievalResult(record=rec, similarity=similarity,
                               relevance_note=f"similar:{quality}")

    def test_build_basic(self):
        ctx = ContextManager(token_budget=2000)
        results = [
            self._make_result("Good XAU trade", "HIGH",   0.90, pnl=3.0),
            self._make_result("Bad EUR trade",  "FALSE_EDGE", 0.80, pnl=-2.0),
        ]
        block = ctx.build(results)
        self.assertGreater(block.memory_count, 0)
        self.assertGreater(block.warning_count, 0)
        self.assertIn("XAU", block.memories_text)
        self.assertIn("EUR", block.warnings_text)

    def test_token_budget_respected(self):
        ctx = ContextManager(token_budget=100)
        results = [self._make_result(f"Memory {i}", "HIGH", 0.9 - i*0.01)
                   for i in range(50)]
        block = ctx.build(results)
        self.assertLessEqual(block.total_tokens, 200)  # some tolerance

    def test_quality_split(self):
        ctx = ContextManager(token_budget=3000)
        results = [
            self._make_result("HIGH quality",   "HIGH",       0.9),
            self._make_result("MEDIUM quality", "MEDIUM",     0.85),
            self._make_result("FALSE_EDGE",     "FALSE_EDGE", 0.8),
            self._make_result("LUCKY",          "LUCKY",      0.75),
        ]
        block = ctx.build(results)
        # HIGH/MEDIUM go to memories_text
        self.assertIn("HIGH quality", block.memories_text)
        # FALSE_EDGE/LUCKY go to warnings_text
        self.assertIn("FALSE_EDGE", block.warnings_text)

    def test_proactive_build(self):
        ctx = ContextManager()
        results = [
            self._make_result("Related past entry", "HIGH", 0.88, pnl=2.5),
        ]
        text = ctx.build_proactive(results)
        self.assertIn("RELATED PAST CONTEXT", text)
        self.assertIn("HIGH", text)


# ---------------------------------------------------------------------------
# SMAClient integration test (offline)
# ---------------------------------------------------------------------------

class TestSMAClientOffline(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sma = SMAClient(
            config=SMAConfig(
                db_path=os.path.join(self.tmpdir, "test.db"),
                qwen_api_key="",
                embed_on_store=False,
            )
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_remember_no_outcome(self):
        mid = self.sma.remember({"content": "Test decision", "domain": "test"})
        self.assertIsNotNone(mid)
        record = self.sma._store.get(mid)
        self.assertEqual(record.quality, "UNCLASSIFIED")

    def test_remember_with_outcome_offline(self):
        mid = self.sma.remember(
            {"content": "Test", "domain": "test",
             "outcome": {"pnl_pct": 2.0}},
            auto_classify=False,   # no API key
        )
        record = self.sma._store.get(mid)
        self.assertIsNotNone(record.outcome)
        self.assertEqual(record.outcome["pnl_pct"], 2.0)

    def test_update_outcome(self):
        mid = self.sma.remember({"content": "Test", "domain": "test"})
        result = self.sma.update_outcome(mid, {"success": True}, auto_classify=False)
        record = self.sma._store.get(mid)
        self.assertEqual(record.outcome, {"success": True})

    def test_get_session(self):
        for i in range(3):
            self.sma.remember({"content": f"Item {i}", "domain": "test"},
                               session_id="sess1")
        records = self.sma.get_session("sess1")
        self.assertEqual(len(records), 3)

    def test_stats(self):
        for i in range(5):
            mid = self.sma.remember({"content": f"Item {i}", "domain": "stats"})
        stats = self.sma.stats(domain="stats")
        self.assertEqual(stats.total, 5)

    def test_recall_no_retriever(self):
        # Without API key, retriever is None — should return empty list gracefully
        self.sma.remember({"content": "XAU trade", "domain": "trading"})
        results = self.sma.recall("XAU trade", domain="trading")
        self.assertIsInstance(results, list)

    def test_context_for_empty(self):
        ctx = self.sma.context_for("some situation")
        self.assertIsInstance(ctx, str)

    def test_pending_classification(self):
        mid = self.sma.remember({"content": "Test", "domain": "test"})
        self.sma._store.update_outcome(mid, {"pnl_pct": 1.0})
        count = self.sma.pending_classification()
        self.assertEqual(count, 1)

    def test_get_high_regret(self):
        mid = self.sma.remember({"content": "Bad trade", "domain": "test"})
        self.sma._store.update_quality(mid, "FALSE_EDGE", 0.9)
        results = self.sma.get_high_regret(domain="test", min_regret=0.7)
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].regret_score, 0.7)

    def test_get_disputed(self):
        mid = self.sma.remember({"content": "Disputed", "domain": "test"})
        self.sma._store.update_quality(mid, "HIGH", 0.1, audit_disputed=True)
        disputed = self.sma.get_disputed()
        self.assertEqual(len(disputed), 1)

    def test_repr(self):
        r = repr(self.sma)
        self.assertIn("SMAClient", r)

    def test_domain_isolation(self):
        """Memories from different domains should not bleed into each other."""
        for domain in ["trading", "legal", "creative"]:
            for i in range(2):
                self.sma.remember({"content": f"{domain} item {i}",
                                   "domain": domain})
        trading = self.sma._store.get_by_domain("trading")
        self.assertEqual(len(trading), 2)
        for r in trading:
            self.assertEqual(r.domain, "trading")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [
        TestMemoryStore,
        TestClassifierOffline,
        TestContextManager,
        TestSMAClientOffline,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
