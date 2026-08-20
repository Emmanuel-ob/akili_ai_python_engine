"""
Structure-aware text chunking.

Engine Revamp Phase 4. Replaces fixed-width character slicing, which cut
documents into 500-character windows with 50 characters of overlap, ignoring
paragraphs, sentences, headings and table rows. Chunks routinely began and
ended mid-word, and an embedding of half a sentence retrieves badly no matter
how good the model is.

Strategy, ported from Trivia's ingestion pipeline where it has survived
production on large textbooks:

  1. Split on paragraph breaks first, which preserves semantic units.
  2. If a paragraph exceeds chunk_size, fall back to sentence boundaries.
  3. Hard-cut only as a last resort, for pathological input with no breaks at
     all: a dense table, an index, an equation dump. Such a chunk would
     otherwise exceed the embedding model's context window and fail on every
     retry, taking the whole document with it.
  4. Coalesce undersized fragments into their neighbours.

Step 4 matters more than it looks. Short paragraphs, headings, labels and list
items get flushed on their own when the next paragraph does not fit, leaving
~30-character chunks. Each would otherwise become its own embedding call and
its own near-useless vector.
"""

from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging_config import logger


def _split_to_ceiling(text: str, chunk_size: int) -> List[str]:
    """Split text into pieces no longer than chunk_size.

    Prefers sentence boundaries. Any single sentence still longer than the
    ceiling is hard-cut into fixed-width windows: this is the last-resort path
    for input that has no usable break anywhere, which must never survive as
    one oversized chunk.
    """
    pieces: List[str] = []
    sub = ""
    for sent in text.replace(". ", ".|").split("|"):
        # A "sentence" longer than the ceiling has no usable break, so cut it
        # into windows before it can be accumulated.
        while len(sent) > chunk_size:
            if sub:
                pieces.append(sub)
                sub = ""
            pieces.append(sent[:chunk_size])
            sent = sent[chunk_size:]
        if not sent:
            continue
        if len(sub) + len(sent) + 1 <= chunk_size:
            sub = (sub + " " + sent).strip() if sub else sent
        else:
            if sub:
                pieces.append(sub)
            sub = sent
    if sub:
        pieces.append(sub)
    return pieces


def _should_merge(
    prev_len: int, cur_len: int, chunk_size: int, min_chunk_size: int
) -> bool:
    """The single coalesce predicate.

    Merge when either side is an undersized fragment and the union still fits
    a soft ceiling (chunk_size + min_chunk_size), so a tiny heading merges into
    a near-full body instead of being stranded alone.
    """
    return (prev_len < min_chunk_size or cur_len < min_chunk_size) and (
        prev_len + cur_len + 2 <= chunk_size + min_chunk_size
    )


class ChunkingService:
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
    ) -> List[str]:
        """Split text into chunks of approximately chunk_size characters.

        Paragraph breaks first, then sentence breaks, then a hard cut. Returns
        non-empty strings longer than 20 characters.

        overlap is accepted for signature compatibility and is unused: natural
        paragraph boundaries provide the context continuity that a sliding
        window was approximating.
        """
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if min_chunk_size is None:
            min_chunk_size = settings.CHUNK_MIN_SIZE

        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > chunk_size:
                    # Too long for one chunk: split by sentence, hard-cutting
                    # any sentence that is itself over the ceiling. All but the
                    # last piece are flushed; the last continues as `current`
                    # so it can still absorb the following paragraph.
                    pieces = _split_to_ceiling(para, chunk_size)
                    for p in pieces[:-1]:
                        chunks.append(p)
                    current = pieces[-1] if pieces else ""
                else:
                    current = para

        if current:
            chunks.append(current)

        coalesced: List[str] = []
        for c in chunks:
            if coalesced and _should_merge(
                len(coalesced[-1]), len(c), chunk_size, min_chunk_size
            ):
                coalesced[-1] = (coalesced[-1] + "\n\n" + c).strip()
            else:
                coalesced.append(c)

        # The minimum-length filter drops debris left over from SPLITTING, so
        # it only applies when splitting actually happened. Trivia can discard
        # anything under 20 characters because it ingests textbooks, where a
        # fragment that short is always noise. This engine ingests FAQs, where
        # "Yes, we deliver on Sundays." is a complete and useful answer, and
        # discarding it would lose the entry entirely.
        if len(coalesced) == 1:
            return [coalesced[0]] if coalesced[0].strip() else []

        return [c for c in coalesced if len(c.strip()) > 20]

    @staticmethod
    def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a batch of documents, preserving every other field.

        A document that yields exactly one chunk keeps its original doc_id.
        Multi-chunk documents get positional ids, `{doc_id}_chunk_{i}`.

        Those ids are positional, which is why the upsert path must delete a
        document's existing chunks before writing new ones: re-chunking into
        fewer pieces would otherwise leave the surplus behind as searchable
        orphans. See routes_embeddings.
        """
        chunked_docs: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("doc_id", "unknown")
            text = doc.get("text")

            if not text or not isinstance(text, str):
                logger.error(f"Invalid or empty text for doc {doc_id}")
                continue

            if len(text.strip()) == 0:
                logger.error(f"Empty text for doc {doc_id}")
                continue

            chunks = ChunkingService.chunk_text(text)
            logger.info(f"Chunked document {doc_id} into {len(chunks)} chunks")

            if not chunks:
                logger.warning(f"Document {doc_id} produced no usable chunks")
                continue

            if len(chunks) == 1:
                single = doc.copy()
                single["text"] = chunks[0]
                chunked_docs.append(single)
                continue

            for i, chunk in enumerate(chunks):
                chunk_doc = doc.copy()
                chunk_doc["text"] = chunk
                chunk_doc["doc_id"] = f"{doc_id}_chunk_{i}"
                chunked_docs.append(chunk_doc)

        logger.info(f"Total documents after chunking: {len(chunked_docs)}")
        return chunked_docs
