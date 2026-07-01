"""
SMA — Retry Utility
====================
Shared retry-with-backoff helper for external API calls (Qwen, DeepSeek).

Why this exists:
  A single transient failure (timeout, momentary rate limit, network blip)
  should not immediately degrade a classification to a fallback value.
  This wraps API calls with bounded retries and exponential backoff,
  and only gives up after genuinely exhausting attempts.

Design:
  - Retries on transient-looking errors (timeouts, connection errors,
    rate limit / 429, 5xx server errors)
  - Does NOT retry on errors that won't be fixed by retrying
    (auth errors, malformed request / 400) — fails fast instead
  - Bounded: max 3 attempts by default, capped total wait time
  - Fully synchronous (matches the rest of SMA's sync OpenAI SDK usage)
"""

import time
from typing import Callable, Optional, Tuple, TypeVar

T = TypeVar("T")

# Error substrings that indicate a transient, retry-worthy failure
_RETRYABLE_MARKERS = (
    "timeout", "timed out", "connection", "rate limit", "429",
    "500", "502", "503", "504", "overloaded", "temporarily",
)

# Error substrings that indicate retrying will NOT help — fail fast
_NON_RETRYABLE_MARKERS = (
    "401", "403", "invalid_api_key", "authentication",
    "400", "invalid request", "does not exist",
)


def _is_retryable(error_text: str) -> bool:
    text = error_text.lower()
    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return False
    return any(marker in text for marker in _RETRYABLE_MARKERS) or True
    # Default to retryable for unknown errors too — better to retry an
    # unrecognized transient issue than to silently degrade quality.


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
) -> Tuple[Optional[T], Optional[str]]:
    """
    Execute fn() with retry + exponential backoff.

    Returns:
        (result, None) on success
        (None, error_message) if all attempts exhausted or error is non-retryable
    """
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return result, None
        except Exception as e:
            last_error = str(e)

            if not _is_retryable(last_error):
                return None, f"non_retryable: {last_error[:150]}"

            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                time.sleep(delay)
                continue

    return None, f"exhausted_{max_attempts}_attempts: {last_error[:150] if last_error else 'unknown'}"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Retry Utility Self-Test ===\n")

    # Test 1: succeeds on first try
    calls = {"count": 0}
    def always_succeeds():
        calls["count"] += 1
        return "ok"
    result, err = with_retry(always_succeeds)
    print(f"✅ Immediate success: result={result}, calls={calls['count']}")
    assert result == "ok" and err is None and calls["count"] == 1

    # Test 2: fails twice (timeout), succeeds on 3rd attempt
    calls = {"count": 0}
    def fails_then_succeeds():
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("Request timed out")
        return "recovered"
    result, err = with_retry(fails_then_succeeds, max_attempts=3, base_delay=0.01)
    print(f"✅ Recovers after retries: result={result}, calls={calls['count']}")
    assert result == "recovered" and calls["count"] == 3

    # Test 3: exhausts all retries on persistent transient error
    calls = {"count": 0}
    def always_times_out():
        calls["count"] += 1
        raise TimeoutError("Connection timed out")
    result, err = with_retry(always_times_out, max_attempts=3, base_delay=0.01)
    print(f"✅ Exhausts retries: result={result}, err='{err}', calls={calls['count']}")
    assert result is None and calls["count"] == 3 and "exhausted" in err

    # Test 4: non-retryable error fails fast (no wasted retries)
    calls = {"count": 0}
    def auth_error():
        calls["count"] += 1
        raise Exception("401 invalid_api_key: authentication failed")
    result, err = with_retry(auth_error, max_attempts=3, base_delay=0.01)
    print(f"✅ Fails fast on auth error: result={result}, calls={calls['count']} (expect 1, not 3)")
    assert result is None and calls["count"] == 1 and "non_retryable" in err

    print("\nretry_utils.py — OK")
