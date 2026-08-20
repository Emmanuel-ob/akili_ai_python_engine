"""
Relevance gating and ordering for retrieved knowledge-base chunks.

Extracted from chat.py during Engine Revamp Phase 1. The logic used to be
inline in a 929-line module that cannot be imported without the whole LLM and
vector stack, which is precisely why two defects survived unnoticed:

  Issue 02  The confidence floor was hardcoded at 0.20 in two call sites
            (chat.py:637 and chat.py:668) while settings.MIN_CONFIDENCE_THRESHOLD
            was declared as 0.5 and never read by anything. At 0.20 cosine
            similarity nearly any chunk qualifies as relevant, so the model was
            handed near-random context and answered confidently from it.

  Issue 03  Hits were re-sorted by the admin-set faq_priority BEFORE the
            confidence check ran, and only results[0] was tested. A high-priority
            but irrelevant chunk could therefore displace the best match and
            become both the thing that opened the gate and the answer.

The ordering rule this module implements:

    1. Discard everything below the relevance floor. Relevance is not
       negotiable and nothing outranks it.
    2. Order the survivors by admin priority, descending. Among chunks that are
       all genuinely relevant, honouring the admin's preference is the point of
       the priority field.
    3. Break ties on score, descending.

The threshold is passed in rather than read from settings here, so this module
stays free of configuration and I/O imports and can be unit-tested with nothing
installed. Callers supply settings.MIN_CONFIDENCE_THRESHOLD.
"""

from typing import Any, List, Optional

# Matches the default used by the vector store payloads. Mid-scale, so a source
# with no explicit priority sits between "deprioritised" and "pinned".
DEFAULT_PRIORITY = 5


def passes_threshold(score: float, threshold: float) -> bool:
    """True when a hit is relevant enough to be shown to the model.

    The boundary is inclusive: a chunk scoring exactly the floor is relevant.
    """
    return score >= threshold


def _priority_of(hit: Any) -> int:
    """Admin-set priority for a hit, defaulting when the metadata is absent.

    Payloads are built by the ingestion pipeline and older records predate the
    faq_priority field, so this must never assume the key exists.
    """
    payload = getattr(hit, "payload", None) or {}
    metadata = payload.get("metadata") or {}
    priority = metadata.get("faq_priority", DEFAULT_PRIORITY)
    try:
        return int(priority)
    except (TypeError, ValueError):
        # A malformed priority must not sink an otherwise relevant chunk.
        return DEFAULT_PRIORITY


def gate_and_order(
    hits: List[Any],
    threshold: float,
    limit: Optional[int] = None,
) -> List[Any]:
    """Filter hits by relevance, then order the survivors.

    Args:
        hits:      Vector store results, each with .score and .payload.
        threshold: Minimum similarity to be considered relevant at all.
        limit:     Optional cap applied after ordering.

    Returns:
        The relevant hits, most preferred first. Empty when nothing clears the
        floor, which the caller should treat as "no usable context" rather than
        falling back to the best of a bad set.
    """
    if not hits:
        return []

    relevant = [h for h in hits if passes_threshold(h.score, threshold)]
    if not relevant:
        return []

    # Priority first, score as the tie-break. Both descending, hence the
    # negation rather than reverse=True, which would invert the tie-break too.
    relevant.sort(key=lambda h: (-_priority_of(h), -h.score))

    return relevant[:limit] if limit is not None else relevant
