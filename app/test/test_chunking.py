"""
Tests for structure-aware chunking.

Run with no engine dependencies installed:
    python -m unittest discover -s app/test -p "test_*.py" -v
"""

import unittest

from app.services.chunking import ChunkingService


class TestChunkText(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(ChunkingService.chunk_text("Hello world."), ["Hello world."])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(ChunkingService.chunk_text(""), [])
        self.assertEqual(ChunkingService.chunk_text("   "), [])

    def test_never_cuts_mid_word_when_sentences_are_available(self):
        sentences = " ".join(f"This is sentence number {i}." for i in range(60))
        chunks = ChunkingService.chunk_text(sentences, chunk_size=200, min_chunk_size=50)
        for c in chunks:
            self.assertFalse(c.startswith(" "))
            # A chunk ending mid-word means the splitter ignored sentence ends.
            self.assertTrue(c.endswith(".") or c.endswith("?") or c.endswith("!"))

    def test_pathological_unbroken_text_is_hard_cut(self):
        # No paragraph or sentence breaks anywhere. Must still be split, or it
        # exceeds the embedding model's context window and fails every retry.
        blob = "X" * 5000
        chunks = ChunkingService.chunk_text(blob, chunk_size=500)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)

    def test_undersized_fragments_are_merged(self):
        # A heading followed by a body should not strand the heading alone.
        text = "Refunds\n\n" + ("Our refund policy is simple. " * 20)
        chunks = ChunkingService.chunk_text(text, chunk_size=600, min_chunk_size=150)
        self.assertNotIn("Refunds", [c.strip() for c in chunks])

    def test_no_chunk_is_trivially_short(self):
        text = "\n\n".join(f"Line {i}." for i in range(40))
        chunks = ChunkingService.chunk_text(text, chunk_size=300, min_chunk_size=100)
        for c in chunks:
            self.assertGreater(len(c.strip()), 20)

    def test_a_short_faq_answer_survives(self):
        # Trivia's chunker discards anything under 20 characters as splitting
        # debris, which is right for textbooks and wrong here: a one-line FAQ
        # answer is the whole entry, and dropping it loses the content.
        self.assertEqual(ChunkingService.chunk_text("Yes."), ["Yes."])
        self.assertEqual(
            ChunkingService.chunk_text("We ship daily."), ["We ship daily."]
        )

    def test_short_debris_is_still_dropped_when_splitting_occurred(self):
        # The filter must still apply where it was earning its keep.
        text = ("A paragraph of reasonable length that will fill a chunk. " * 6) + "\n\nx"
        chunks = ChunkingService.chunk_text(text, chunk_size=200, min_chunk_size=50)
        self.assertNotIn("x", [c.strip() for c in chunks])

    def test_produces_fewer_chunks_than_fixed_width(self):
        # The regression this phase exists to fix: fixed-width slicing of
        # well-structured prose over-fragments it.
        text = "\n\n".join(
            "This is a paragraph with several sentences. " * 3 for _ in range(10)
        )
        old_count = len(range(0, len(text), 450))  # what 500/50 slicing produced
        new = ChunkingService.chunk_text(text, chunk_size=500, min_chunk_size=150)
        self.assertLessEqual(len(new), old_count)


class TestChunkDocuments(unittest.TestCase):
    def test_single_chunk_document_keeps_its_doc_id(self):
        docs = [{"doc_id": "abc", "text": "Short."}]
        out = ChunkingService.chunk_documents(docs)
        self.assertEqual(out[0]["doc_id"], "abc")

    def test_multi_chunk_document_gets_indexed_ids(self):
        docs = [{"doc_id": "abc", "text": "Sentence. " * 400}]
        out = ChunkingService.chunk_documents(docs)
        self.assertGreater(len(out), 1)
        self.assertEqual(out[0]["doc_id"], "abc_chunk_0")
        self.assertEqual(out[1]["doc_id"], "abc_chunk_1")

    def test_metadata_is_carried_onto_every_chunk(self):
        docs = [
            {
                "doc_id": "abc",
                "text": "Sentence. " * 400,
                "metadata": {"source_type": "faq", "faq_priority": 9},
            }
        ]
        out = ChunkingService.chunk_documents(docs)
        for chunk in out:
            self.assertEqual(chunk["metadata"]["faq_priority"], 9)

    def test_empty_document_is_skipped_not_crashed(self):
        docs = [
            {"doc_id": "a", "text": ""},
            {"doc_id": "b", "text": "Real content here."},
        ]
        out = ChunkingService.chunk_documents(docs)
        self.assertEqual([d["doc_id"] for d in out], ["b"])

    def test_non_string_text_is_skipped(self):
        docs = [{"doc_id": "a", "text": None}, {"doc_id": "b", "text": "Real."}]
        out = ChunkingService.chunk_documents(docs)
        self.assertEqual([d["doc_id"] for d in out], ["b"])


class TestOrphanChunkContract(unittest.TestCase):
    """Chunk ids are positional, so a re-chunk that produces fewer pieces
    leaves the surplus behind unless they are deleted first."""

    def test_rechunking_smaller_produces_a_shorter_id_range(self):
        long_text = "Sentence. " * 400
        many = ChunkingService.chunk_documents([{"doc_id": "abc", "text": long_text}])
        few = ChunkingService.chunk_documents(
            [{"doc_id": "abc", "text": "Sentence. " * 40}]
        )
        self.assertGreater(len(many), len(few))
        stale = {d["doc_id"] for d in many} - {d["doc_id"] for d in few}
        # These ids would survive an upsert keyed on doc_id.
        self.assertTrue(stale)


if __name__ == "__main__":
    unittest.main()
