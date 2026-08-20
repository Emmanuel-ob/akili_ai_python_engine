"""
Resilience helpers for the embedding client.

Engine Revamp Phase 4, charter section 0.2. The client previously sent an
entire document's chunks as ONE request, with no retry, no backoff and no
input cap, and raised on any failure. So a single oversized chunk killed the
whole ingestion, and a transient 503 lost a document that had almost finished.

Trivia hit this on a large physics textbook that died at roughly 80% and left
the diagnosis in a comment. These are the same four patterns, extracted as
pure functions with no SDK imports so they are unit-testable on a machine
with nothing installed.

The four:

  cap_input               a hard character ceiling, so one pathological chunk
                          cannot exceed the model's context window
  batch_texts             bounded groups, so one failure loses a batch rather
                          than a document
  is_context_length_error tells a "too long" rejection apart from a transient
                          failure, because they need opposite responses
  with_retry              exponential backoff for transient failures, and
                          halve-and-retry for over-length input

That last distinction is the subtle one. Backing off and re-sending identical
over-length text fails identically every time, however long you wait. The only
useful response is to shrink the input.
"""

import time
from typing import Any, Callable, Iterator, List, Sequence, TypeVar

from app.core.logging_config import logger

T = TypeVar("T")

# Below this, shrinking further is pointless: if a few hundred characters
# still trip a context-length error, the problem is not the length.
_MIN_SHRINK_CHARS = 500


def cap_input(text: str, max_chars: int) -> str:
    """Truncate text to the embedding model's safe input ceiling.

    Only the vector input is capped. The FULL chunk is still stored and
    returned to callers, so nothing is lost from the answer text; only the
    retrieval vector sees the truncated head.
    """
    if not text:
        return text
    return text[:max_chars]


def batch_texts(texts: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield bounded batches, preserving order."""
    if not texts:
        return
    step = max(1, size)
    for start in range(0, len(texts), step):
        yield list(texts[start : start + step])


def is_context_length_error(err: Exception) -> bool:
    """True when the endpoint rejected the input for being too long.

    Distinct from a transient 5xx or timeout, which should be retried
    unchanged. Providers phrase this differently, so match on the shared
    vocabulary rather than a status code.
    """
    parts = [str(err)]
    response = getattr(err, "response", None)
    if response is not None:
        parts.append(str(getattr(response, "text", "")))
    blob = " ".join(parts).lower()

    return any(
        phrase in blob
        for phrase in (
            "context length",
            "exceeds the context",
            "token count exceeds",
            "input is too long",
            "maximum context",
        )
    )


def with_retry(
    fn: Callable[[Any], Any],
    text: Any,
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
    shrink_on_overlength: bool = True,
) -> Any:
    """Call fn(text), retrying on failure.

    Two failure modes, handled differently:

      over-length  shrink the input by half and retry IMMEDIATELY. No sleep,
                   because waiting changes nothing about the length.
      transient    back off exponentially (1s, 2s, 4s) and retry unchanged.

    Shrinking applies to a STRING payload only. A batch (a list of texts) must
    never be sliced on an over-length error: that would silently drop texts and
    return fewer embeddings than the caller passed in, which is data loss
    rather than a failure. Pass shrink_on_overlength=False for batch calls, or
    rely on the isinstance guard below.

    sleep is injectable so tests run instantly.

    Raises the last exception once retries are exhausted, so the caller can
    record which chunk failed rather than silently storing nothing.
    """
    payload = text
    last_error: Exception = RuntimeError("no attempt was made")
    attempt = 0
    backoff_step = 0

    while attempt < retries:
        attempt += 1
        try:
            return fn(payload)
        except Exception as err:
            last_error = err

            if (
                shrink_on_overlength
                and is_context_length_error(err)
                and isinstance(payload, str)
                and len(payload) > _MIN_SHRINK_CHARS
            ):
                new_len = len(payload) // 2
                logger.warning(
                    f"Embedding input too long ({len(payload)} chars); "
                    f"retrying at {new_len}"
                )
                payload = payload[:new_len]
                # Shrinking is deterministic, so this attempt is not "used up"
                # in the same sense a transient failure is: keep the backoff
                # budget for genuine flakiness.
                attempt -= 1
                retries -= 1
                if retries <= 0:
                    break
                continue

            if attempt < retries:
                wait = 2**backoff_step
                backoff_step += 1
                logger.warning(
                    f"Embedding call failed ({err}); retrying in {wait}s "
                    f"[{attempt}/{retries}]"
                )
                sleep(wait)

    raise last_error
