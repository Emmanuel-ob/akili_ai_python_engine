"""
Chunk id parsing and grouping.

Positional chunk ids look like "{source_id}_chunk_{index}". Recovering the
source id from one is deceptively easy to get wrong, because production ids
contain the separator inside the source portion:

    6978e3e4...:6978e3f9...:my_portfolio.users:1_chunk_0
                                            ^^         ^^^^^^^^
                                    part of the id      the real suffix

Splitting on the FIRST occurrence truncates the id. Splitting on the last,
and only when the suffix is an index, is what works.

Lives in its own module with no SDK imports so it can be unit-tested without
the embedding stack installed. That separation is the same reason
retrieval_gate.py and llm_provider.py exist: logic buried inside a module
that cannot be imported is logic that never gets tested.
"""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

CHUNK_MARKER = "_chunk_"


def source_id_of(doc_id: str) -> str:
    """Recover the source document id from a chunk id.

    Returns doc_id unchanged when it is not a positional chunk id.
    """
    if CHUNK_MARKER not in doc_id:
        return doc_id
    head, _, tail = doc_id.rpartition(CHUNK_MARKER)
    return head if tail.isdigit() else doc_id


def chunk_index_of(doc_id: str) -> int:
    """Positional index of a chunk id, or 0 when it is not chunked."""
    _, _, tail = doc_id.rpartition(CHUNK_MARKER)
    return int(tail) if tail.isdigit() else 0


# Below this length a shared region between two chunks is coincidence rather
# than a real overlap, and removing it would delete genuine text.
_MIN_OVERLAP = 8


def reassemble(chunks: Sequence[str]) -> str:
    """Rebuild a document's text from its stored chunks.

    The old chunker produced fixed-width windows with CHUNK_OVERLAP characters
    of overlap, so adjacent chunks share text. Two naive joins are both wrong:

      "".join(...)      duplicates the shared region
      "\\n\\n".join(...)  inserts a paragraph break exactly where the old
                        chunker cut, often mid-word. The new paragraph-first
                        chunker then splits there, so the old bad boundary
                        survives the re-embed and nothing improves.

    Instead, find the longest suffix of one chunk that is a prefix of the next
    and drop the duplicate. Detecting the overlap rather than assuming
    CHUNK_OVERLAP means this still works for documents chunked under a
    different setting, which matters because the value is configurable.
    """
    parts = [c for c in chunks if c]
    if not parts:
        return ""

    out = parts[0]
    for nxt in parts[1:]:
        limit = min(len(out), len(nxt))
        overlap = 0
        # Longest first: a short match inside a longer one is not the boundary.
        for size in range(limit, _MIN_OVERLAP - 1, -1):
            if out[-size:] == nxt[:size]:
                overlap = size
                break
        out += nxt[overlap:]

    return out


def group_by_source(rows: Sequence[Tuple[Any, ...]]) -> Dict[str, List[Tuple]]:
    """Group chunk rows back into the documents they came from.

    Rows are (doc_id, ...) tuples; only the first element is inspected.

    Chunks are ordered by their numeric index rather than lexically, because
    lexical order puts _chunk_10 before _chunk_2 and would reassemble the
    document's text out of sequence.
    """
    grouped: Dict[str, List[Tuple]] = defaultdict(list)
    for row in rows:
        grouped[source_id_of(row[0])].append(row)

    for source in grouped:
        grouped[source].sort(key=lambda r: chunk_index_of(r[0]))

    return dict(grouped)
