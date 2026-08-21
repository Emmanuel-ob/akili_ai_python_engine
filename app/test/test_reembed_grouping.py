"""
Tests for re-embed chunk grouping.

The parsing here is easy to get wrong because production doc_ids contain the
separator inside the source id itself:

    6978e3e4...:6978e3f9...:my_portfolio.users:1_chunk_0
                                            ^^         ^^^^^^^^
                                     part of the id     the real suffix

Splitting on the FIRST "_chunk_" would truncate the source id. Splitting on
the last one, and only when the suffix is an index, is what works.

Getting this wrong would regroup chunks under the wrong source and rebuild
documents from other documents' text, so it is worth pinning.
"""

import unittest

from app.services.chunk_ids import chunk_index_of, group_by_source, source_id_of

REAL = "6978e3e427622c8d2309b394:6978e3f927622c8d2309b398:my_portfolio.users:1"


class TestSourceIdOf(unittest.TestCase):
    def test_recovers_the_source_from_a_chunk_id(self):
        self.assertEqual(source_id_of(f"{REAL}_chunk_0"), REAL)

    def test_recovers_from_a_multi_digit_index(self):
        self.assertEqual(source_id_of(f"{REAL}_chunk_137"), REAL)

    def test_an_unchunked_id_is_returned_unchanged(self):
        self.assertEqual(source_id_of(REAL), REAL)

    def test_a_source_id_containing_the_marker_is_not_truncated(self):
        # "_chunk_" appearing inside the document's own name must not be
        # mistaken for the positional suffix.
        tricky = "biz:bot:my_chunk_notes.pdf:3"
        self.assertEqual(source_id_of(f"{tricky}_chunk_2"), tricky)

    def test_a_non_numeric_suffix_is_not_treated_as_an_index(self):
        weird = "biz:bot:doc_chunk_final"
        self.assertEqual(source_id_of(weird), weird)


class TestGroupBySource(unittest.TestCase):
    @staticmethod
    def _row(doc_id, text="t", meta=None):
        return (doc_id, "biz", "bot", text, meta or {})

    def test_groups_chunks_under_their_source(self):
        rows = [self._row(f"{REAL}_chunk_{i}") for i in range(3)]
        grouped = group_by_source(rows)
        self.assertEqual(list(grouped), [REAL])
        self.assertEqual(len(grouped[REAL]), 3)

    def test_separates_different_sources(self):
        other = "biz:bot:other.doc:9"
        rows = [self._row(f"{REAL}_chunk_0"), self._row(f"{other}_chunk_0")]
        self.assertEqual(len(group_by_source(rows)), 2)

    def test_orders_chunks_numerically_not_lexically(self):
        # Lexical order puts _chunk_10 before _chunk_2, which would reassemble
        # the document's text in the wrong order.
        rows = [self._row(f"{REAL}_chunk_{i}", text=str(i)) for i in (0, 1, 2, 10, 11)]
        rows.reverse()
        ordered = [r[3] for r in group_by_source(rows)[REAL]]
        self.assertEqual(ordered, ["0", "1", "2", "10", "11"])

    def test_an_unchunked_document_groups_under_itself(self):
        grouped = group_by_source([self._row(REAL)])
        self.assertEqual(list(grouped), [REAL])


if __name__ == "__main__":
    unittest.main()
