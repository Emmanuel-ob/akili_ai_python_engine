"""
Tests for the chunk-cleanup LIKE pattern.

Real doc_ids from production look like:

    6978e3e4...:6978e3f9...:my_portfolio.users:1_chunk_0

They contain colons, dots and underscores. In SQL LIKE, `_` is a
single-character wildcard, so an unescaped pattern built from such an id would
also match a DIFFERENT document and delete its chunks. These tests pin the
escaping that prevents that.

Pure string logic, no database required:
    python -m unittest discover -s app/test -p "test_*.py" -v
"""

import re
import unittest

from app.services.vectorstore import VectorStoreService


def like_matches(pattern: str, value: str, escape: str = "!") -> bool:
    """Evaluate a SQL LIKE pattern with ESCAPE in pure Python.

    Mirrors Postgres semantics closely enough to prove the escaping works:
    `%` matches any run, `_` matches exactly one character, and a character
    preceded by the escape char is literal.
    """
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == escape and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.fullmatch("".join(out), value) is not None


REAL_ID = "6978e3e427622c8d2309b394:6978e3f927622c8d2309b398:my_portfolio.users:1"


class TestChunkPatternFor(unittest.TestCase):
    def test_matches_its_own_chunks(self):
        pattern = VectorStoreService.chunk_pattern_for(REAL_ID)
        for i in range(3):
            self.assertTrue(like_matches(pattern, f"{REAL_ID}_chunk_{i}"))

    def test_does_not_match_a_lookalike_document(self):
        # The wildcard bug: unescaped, `my_portfolio` also matches
        # `myXportfolio`, so another document's chunks would be deleted.
        pattern = VectorStoreService.chunk_pattern_for(REAL_ID)
        lookalike = REAL_ID.replace("my_portfolio", "myXportfolio")
        self.assertFalse(like_matches(pattern, f"{lookalike}_chunk_0"))

    def test_does_not_match_a_different_document_entirely(self):
        pattern = VectorStoreService.chunk_pattern_for(REAL_ID)
        self.assertFalse(like_matches(pattern, "someone:else:doc:9_chunk_0"))

    def test_does_not_match_the_unchunked_base_row(self):
        # A single-chunk document keeps its bare doc_id and must survive.
        pattern = VectorStoreService.chunk_pattern_for(REAL_ID)
        self.assertFalse(like_matches(pattern, REAL_ID))

    def test_percent_in_a_doc_id_is_escaped(self):
        doc_id = "biz:bot:report_100%_complete:1"
        pattern = VectorStoreService.chunk_pattern_for(doc_id)
        self.assertTrue(like_matches(pattern, f"{doc_id}_chunk_0"))
        self.assertFalse(
            like_matches(pattern, "biz:bot:report_100ANYTHING_complete:1_chunk_0")
        )

    def test_literal_escape_char_in_a_doc_id_is_itself_escaped(self):
        doc_id = "biz:bot:od!d_name:1"
        pattern = VectorStoreService.chunk_pattern_for(doc_id)
        self.assertTrue(like_matches(pattern, f"{doc_id}_chunk_0"))

    def test_matches_multi_digit_chunk_indexes(self):
        pattern = VectorStoreService.chunk_pattern_for(REAL_ID)
        self.assertTrue(like_matches(pattern, f"{REAL_ID}_chunk_42"))


if __name__ == "__main__":
    unittest.main()
