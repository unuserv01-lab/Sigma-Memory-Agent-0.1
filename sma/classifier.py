"""
SMA — Memory Classifier
=======================
Two-model quality classification pipeline:

  PRIMARY:  Qwen3.7-plus  → classify decision quality + regret score
  AUDITOR:  DeepSeek-chat → independently audit Qwen's assessment
  OUTCOME:  Consensus logic → HIGH confidence if both agree,
            DISPUTED flag if they significantly disagree

Why two models?
  Single-model classification risks systematic bias — especially
  for LUCKY vs HIGH (model may not detect lucky patterns) and
  FALSE_EDGE (model may not penalize confident-but-wrong).
  DeepSeek as independent auditor catches these cases without
  requiring human review for every entry. Disputed entries are
  flagged for optional human review.

Developer extension points:
  regret_fn:     inject domain-specific regret calculation
  quality_rubric: inject rule-based classification (bypasses LLM for primary)
  audit_enabled: toggle DeepSeek audit (default True if API key provided)
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class QualityLevel(str, Enum):
    HIGH        = "HIGH"          # Clear signal, correct reasoning, good outcome
    MEDIUM      = "MEDIUM"        # Reasonable decision, acceptable outcome
    LOW         = "LOW"           # Weak signal or poor reasoning, bad outcome
    LUCKY       = "LUCKY"         # Good outcome, wrong reasoning — dangerous pattern
    FALSE_EDGE  = "FALSE_EDGE"    # High confidence, wrong outcome — most dangerous
    UNCLASSIFIED = "UNCLASSIFIED" # No outcome yet


# Levels ordered by learning value (higher = more trustworthy to learn from)
QUALITY_TRUST_ORDER: Dict[str, int] = {
    "HIGH": 5, "MEDIUM": 4, "LOW": 3, "LUCKY": 2, "FALSE_EDGE": 1, "UNCLASSIFIED": 0
}

RegretFn = Callable[[Dict[str, Any]], float]
QualityRubric = Dict[str, Callable[[Dict[str, Any]], bool]]


@dataclass
class QualityResult:
    classification: str           # QualityLevel value
    regret_score: float           # 0.0 → 1.0
    qwen_classification: str
    qwen_reasoning: str
    deepseek_verdict: Optional[str]      # AGREE / DISAGREE / SKIPPED
    deepseek_classification: Optional[str]
    deepseek_reasoning: Optional[str]
    consensus: bool
    disputed: bool
    confidence: float             # 0.0 → 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "regret_score": self.regret_score,
            "qwen_classification": self.qwen_classification,
            "qwen_reasoning": self.qwen_reasoning,
            "deepseek_verdict": self.deepseek_verdict,
            "deepseek_classification": self.deepseek_classification,
            "deepseek_reasoning": self.deepseek_reasoning,
            "consensus": self.consensus,
            "disputed": self.disputed,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MemoryClassifier:
    """
    Classifies agent decision quality using Qwen + optional DeepSeek audit.

    Usage:
        classifier = MemoryClassifier(
            qwen_api_key="sk-...",
            deepseek_api_key="sk-...",   # optional
        )
        result = classifier.classify(entry)

    Injectable extensions:
        MemoryClassifier(
            qwen_api_key="sk-...",
            regret_fn=lambda e: max(0, -e["outcome"]["pnl_pct"] / 5.0),
            quality_rubric={
                "HIGH":   lambda e: e["outcome"]["pnl_pct"] > 2.0,
                "MEDIUM": lambda e: 0 < e["outcome"]["pnl_pct"] <= 2.0,
                "LOW":    lambda e: e["outcome"]["pnl_pct"] <= 0,
                "LUCKY":  lambda e: e["outcome"]["pnl_pct"] > 0
                                    and not e.get("signal_followed"),
                "FALSE_EDGE": lambda e: e["outcome"]["pnl_pct"] < 0
                                        and e.get("confidence", 0) > 0.7,
            }
        )
    """

    QWEN_BASE_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    QWEN_MODEL        = "qwen3.7-plus"
    DEEPSEEK_MODEL    = "deepseek-chat"

    def __init__(
        self,
        qwen_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        regret_fn: Optional[RegretFn] = None,
        quality_rubric: Optional[QualityRubric] = None,
        audit_rubric: bool = False,    # audit even when rubric provided
        temperature: float = 0.1,      # low temp for deterministic classification
    ):
        self.qwen_api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.regret_fn = regret_fn
        self.quality_rubric = quality_rubric
        self.audit_rubric = audit_rubric
        self.temperature = temperature

        if not self.qwen_api_key:
            raise ValueError("DASHSCOPE_API_KEY required for classification")

        self._qwen = OpenAI(
            api_key=self.qwen_api_key,
            base_url=self.QWEN_BASE_URL,
        )
        self._deepseek: Optional[OpenAI] = None
        if self.deepseek_api_key:
            self._deepseek = OpenAI(
                api_key=self.deepseek_api_key,
                base_url=self.DEEPSEEK_BASE_URL,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, entry: Dict[str, Any]) -> QualityResult:
        """
        Classify a decision entry.
        entry should contain at minimum: 'content' and 'outcome'.
        Everything else in entry is passed as context.
        """
        if not entry.get("outcome"):
            return self._unclassified_result("no_outcome_provided")

        # Step 1: Primary classification
        if self.quality_rubric:
            qwen_class, qwen_reason = self._rubric_classify(entry)
            skip_audit = not self.audit_rubric
        else:
            qwen_class, qwen_reason = self._qwen_classify(entry)
            skip_audit = False

        # Step 2: Regret score
        regret = self._compute_regret(entry, qwen_class)

        # Step 3: DeepSeek audit (if available and not explicitly skipped)
        ds_verdict = ds_class = ds_reason = None
        if self._deepseek and not skip_audit:
            ds_verdict, ds_class, ds_reason = self._deepseek_audit(
                entry, qwen_class, qwen_reason, regret
            )

        # Step 4: Consensus resolution
        final_class, consensus, disputed, confidence = self._resolve(
            qwen_class, ds_verdict, ds_class
        )

        return QualityResult(
            classification=final_class,
            regret_score=round(regret, 4),
            qwen_classification=qwen_class,
            qwen_reasoning=qwen_reason,
            deepseek_verdict=ds_verdict,
            deepseek_classification=ds_class,
            deepseek_reasoning=ds_reason,
            consensus=consensus,
            disputed=disputed,
            confidence=round(confidence, 3),
        )

    # ------------------------------------------------------------------
    # Step 1: Primary classification
    # ------------------------------------------------------------------

    def _qwen_classify(self, entry: Dict[str, Any]):
        prompt = self._build_classification_prompt(entry)
        try:
            resp = self._qwen.chat.completions.create(
                model=self.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": self._classification_system_prompt()},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse_classification_response(raw)
        except Exception as e:
            return "MEDIUM", f"qwen_error: {str(e)[:100]}"

    def _rubric_classify(self, entry: Dict[str, Any]):
        """Evaluate developer-provided rule-based rubric.
        Order matters: specific patterns (FALSE_EDGE, LUCKY) checked before
        general ones (MEDIUM, LOW) to prevent early misclassification.
        """
        for level in ["HIGH", "FALSE_EDGE", "LUCKY", "MEDIUM", "LOW"]:
            fn = self.quality_rubric.get(level)
            if fn:
                try:
                    if fn(entry):
                        return level, "rubric_match"
                except Exception:
                    continue
        return "LOW", "no_rubric_match"

    # ------------------------------------------------------------------
    # Step 2: Regret score
    # ------------------------------------------------------------------

    def _compute_regret(self, entry: Dict[str, Any], classification: str) -> float:
        """
        Compute regret score.
        Priority: injectable regret_fn → outcome-based fallback → classification-based fallback
        """
        # Developer-injected function
        if self.regret_fn:
            try:
                score = float(self.regret_fn(entry))
                return max(0.0, min(1.0, score))
            except Exception:
                pass

        # Outcome-based fallback (works for any domain with numeric outcome)
        outcome = entry.get("outcome", {})

        # Trading-specific
        if "pnl_pct" in outcome:
            pnl = float(outcome["pnl_pct"])
            if pnl >= 2.0:  return 0.0
            if pnl >= 0:    return max(0.0, 0.3 - pnl * 0.15)
            if pnl >= -1.0: return 0.5
            return min(1.0, 0.5 + abs(pnl) * 0.1)

        # Generic success/failure
        if "success" in outcome or "correct" in outcome:
            success = outcome.get("success", outcome.get("correct", False))
            if success:
                return 0.1 if classification in ("HIGH", "MEDIUM") else 0.5
            else:
                return 0.8 if classification in ("HIGH", "MEDIUM") else 0.5

        # Classification-based fallback
        fallback = {
            "HIGH": 0.05, "MEDIUM": 0.25, "LOW": 0.6,
            "LUCKY": 0.55, "FALSE_EDGE": 0.9
        }
        return fallback.get(classification, 0.5)

    # ------------------------------------------------------------------
    # Step 3: DeepSeek audit
    # ------------------------------------------------------------------

    def _deepseek_audit(
        self,
        entry: Dict[str, Any],
        qwen_class: str,
        qwen_reason: str,
        regret_score: float,
    ):
        prompt = self._build_audit_prompt(entry, qwen_class, qwen_reason, regret_score)
        try:
            resp = self._deepseek.chat.completions.create(
                model=self.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": self._audit_system_prompt()},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse_audit_response(raw)
        except Exception as e:
            return "SKIPPED", qwen_class, f"deepseek_error: {str(e)[:100]}"

    # ------------------------------------------------------------------
    # Step 4: Consensus resolution
    # ------------------------------------------------------------------

    def _resolve(self, qwen_class: str, ds_verdict: Optional[str],
                 ds_class: Optional[str]):
        """
        Determine final classification from both models.

        Rules:
          - No DeepSeek → use Qwen, confidence 0.7
          - DeepSeek AGREE → consensus, confidence 0.95
          - DeepSeek DISAGREE (minor, adjacent level) → use Qwen, confidence 0.75, no dispute
          - DeepSeek DISAGREE (major, e.g. HIGH vs FALSE_EDGE) → DISPUTED, use conservative, confidence 0.4
          - DeepSeek SKIPPED → use Qwen, confidence 0.7
        """
        if ds_verdict is None or ds_verdict == "SKIPPED":
            return qwen_class, False, False, 0.7

        if ds_verdict == "AGREE":
            return qwen_class, True, False, 0.95

        # Disagreement
        q_trust = QUALITY_TRUST_ORDER.get(qwen_class, 0)
        d_trust = QUALITY_TRUST_ORDER.get(ds_class or "", 0)
        delta = abs(q_trust - d_trust)

        if delta <= 1:  # Adjacent levels — minor disagreement
            return qwen_class, False, False, 0.75

        # Major disagreement — flag as disputed, use conservative (lower trust level)
        conservative = qwen_class if q_trust <= d_trust else (ds_class or qwen_class)
        return conservative, False, True, 0.4

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _classification_system_prompt() -> str:
        return """You are a precise decision quality classifier for AI agent memory systems.
Classify each decision entry into exactly one of:
- HIGH: Clear signal, sound reasoning, good outcome. Safe to learn from.
- MEDIUM: Reasonable decision, acceptable outcome. Cautious learning.
- LOW: Weak signal or poor reasoning, bad outcome. Avoid this pattern.
- LUCKY: Good outcome but the reasoning was flawed or the signal was weak. DANGEROUS — agent got lucky. Do not reinforce.
- FALSE_EDGE: High confidence in reasoning but bad outcome. MOST DANGEROUS — agent was confident and wrong.

Always respond with valid JSON only. No markdown, no preamble."""

    @staticmethod
    def _audit_system_prompt() -> str:
        return """You are an independent auditor of AI agent decision quality assessments.
Your role is to catch bias in the primary classifier — especially:
1. Confusing LUCKY patterns with HIGH (good outcome ≠ good decision)
2. Missing FALSE_EDGE (high confidence wrong decisions are the most dangerous)
3. Being too lenient with LOW decisions

Always respond with valid JSON only. No markdown, no preamble."""

    def _build_classification_prompt(self, entry: Dict[str, Any]) -> str:
        # Exclude embeddings from prompt (too large and not useful)
        clean_entry = {k: v for k, v in entry.items() if k != "embedding"}
        return f"""Classify this agent decision:

{json.dumps(clean_entry, indent=2, default=str)}

Respond in JSON:
{{
  "classification": "HIGH|MEDIUM|LOW|LUCKY|FALSE_EDGE",
  "reasoning": "one sentence explanation"
}}"""

    def _build_audit_prompt(
        self,
        entry: Dict[str, Any],
        qwen_class: str,
        qwen_reason: str,
        regret_score: float,
    ) -> str:
        clean_entry = {k: v for k, v in entry.items() if k != "embedding"}
        return f"""Audit this decision quality assessment:

DECISION:
{json.dumps(clean_entry, indent=2, default=str)}

PRIMARY ASSESSMENT:
Classification: {qwen_class}
Reasoning: {qwen_reason}
Regret Score: {regret_score:.3f}

Do you AGREE or DISAGREE? If DISAGREE, provide your classification.

Respond in JSON:
{{
  "verdict": "AGREE|DISAGREE",
  "classification": "HIGH|MEDIUM|LOW|LUCKY|FALSE_EDGE",
  "reasoning": "one sentence — focus on what primary classifier may have missed"
}}"""

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_classification_response(raw: str):
        try:
            clean = raw.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(clean)
            classification = data.get("classification", "MEDIUM").upper()
            if classification not in ("HIGH", "MEDIUM", "LOW", "LUCKY", "FALSE_EDGE"):
                classification = "MEDIUM"
            return classification, data.get("reasoning", "")
        except Exception:
            return "MEDIUM", f"parse_error: {raw[:100]}"

    @staticmethod
    def _parse_audit_response(raw: str):
        try:
            clean = raw.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(clean)
            verdict = data.get("verdict", "SKIPPED").upper()
            if verdict not in ("AGREE", "DISAGREE"):
                verdict = "SKIPPED"
            classification = data.get("classification", "MEDIUM").upper()
            if classification not in ("HIGH", "MEDIUM", "LOW", "LUCKY", "FALSE_EDGE"):
                classification = "MEDIUM"
            return verdict, classification, data.get("reasoning", "")
        except Exception:
            return "SKIPPED", "MEDIUM", f"audit_parse_error: {raw[:100]}"

    @staticmethod
    def _unclassified_result(reason: str) -> QualityResult:
        return QualityResult(
            classification="UNCLASSIFIED",
            regret_score=-1.0,
            qwen_classification="UNCLASSIFIED",
            qwen_reasoning=reason,
            deepseek_verdict=None,
            deepseek_classification=None,
            deepseek_reasoning=None,
            consensus=False,
            disputed=False,
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Self-test (offline — verifies logic without API)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test consensus resolution logic without API calls
    classifier = MemoryClassifier.__new__(MemoryClassifier)
    classifier._deepseek = None

    # Test rubric inject
    classifier.quality_rubric = {
        "HIGH":       lambda e: e.get("outcome", {}).get("pnl_pct", 0) > 2.0,
        "MEDIUM":     lambda e: 0 < e.get("outcome", {}).get("pnl_pct", 0) <= 2.0,
        "LOW":        lambda e: e.get("outcome", {}).get("pnl_pct", 0) < 0,
        "LUCKY":      lambda e: (e.get("outcome", {}).get("pnl_pct", 0) > 0
                                 and not e.get("signal_followed", True)),
        "FALSE_EDGE": lambda e: (e.get("outcome", {}).get("pnl_pct", 0) < 0
                                 and e.get("confidence", 0) > 0.7),
    }
    classifier.regret_fn = None
    classifier.audit_rubric = False

    print("=== Classifier self-test (offline) ===\n")

    cases = [
        {"content": "LONG XAU", "outcome": {"pnl_pct": 3.2}, "signal_followed": True, "confidence": 0.8},
        {"content": "SHORT EUR", "outcome": {"pnl_pct": 1.5}, "signal_followed": True},
        {"content": "LONG BTC",  "outcome": {"pnl_pct": -2.1}, "confidence": 0.75},
        {"content": "SHORT GBP", "outcome": {"pnl_pct": 1.0}, "signal_followed": False},
    ]

    expected = ["HIGH", "MEDIUM", "FALSE_EDGE", "LUCKY"]

    for case, exp in zip(cases, expected):
        cls_name, reason = classifier._rubric_classify(case)
        regret = classifier._compute_regret(case, cls_name)
        ok = "✅" if cls_name == exp else "❌"
        print(f"{ok} {cls_name:12} (expected {exp:12}) | regret={regret:.2f} | {case['content']}")

    # Consensus tests
    print("\n--- Consensus resolution ---")
    cases_c = [
        ("HIGH", "AGREE",    "HIGH",       True,  False, 0.95),
        ("HIGH", "DISAGREE", "MEDIUM",     False, False, 0.75),
        ("HIGH", "DISAGREE", "FALSE_EDGE", False, True,  0.4),
        ("HIGH", "SKIPPED",  None,         False, False, 0.7),
    ]
    for q, ds_v, ds_c, exp_cons, exp_disp, exp_conf in cases_c:
        final, cons, disp, conf = classifier._resolve(q, ds_v, ds_c)
        ok = "✅" if (cons == exp_cons and disp == exp_disp) else "❌"
        print(f"{ok} qwen={q} ds={ds_v}/{ds_c} → final={final} cons={cons} disp={disp} conf={conf}")

    print("\nclassifier.py — OK")
