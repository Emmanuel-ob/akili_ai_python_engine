"""
Tests for reassembling a document from its stored chunks.

The old chunker cut fixed 500-character windows with 50 characters of
OVERLAP, so adjacent chunks share text:

    chunk_0  '...I focus on delivering MVPs that solve real pro'
    chunk_1  'xt, I focus on delivering MVPs that solve rea...'

Naive joining is wrong in two ways at once. Concatenating duplicates the
shared region, and joining with "\\n\\n" inserts a paragraph break exactly
where the old chunker cut mid-word, which the new paragraph-first chunker
then faithfully splits on. Either way the old bad boundaries survive the
re-embed, defeating the point of it.
"""

import unittest

from app.services.chunk_ids import reassemble


class TestReassemble(unittest.TestCase):
    def test_single_chunk_is_returned_unchanged(self):
        self.assertEqual(reassemble(["Just the one."]), "Just the one.")

    def test_empty_input(self):
        self.assertEqual(reassemble([]), "")

    def test_removes_the_overlapping_region(self):
        a = "The quick brown fox jumps over the lazy dog"
        b = "over the lazy dog and keeps running"
        self.assertEqual(
            reassemble([a, b]),
            "The quick brown fox jumps over the lazy dog and keeps running",
        )

    def test_reconstructs_a_word_split_across_chunks(self):
        # The real failure: "with" was cut into "wi" + "th".
        a = "I build things wi"
        b = "th a strong foundation in Laravel"
        self.assertEqual(reassemble([a, b]), "I build things with a strong foundation in Laravel")

    def test_handles_three_chunks(self):
        parts = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"]
        self.assertEqual(reassemble(parts), "alpha beta gamma delta epsilon")

    def test_no_overlap_concatenates_directly(self):
        # Nothing shared, so nothing to remove and nothing to invent.
        self.assertEqual(reassemble(["abc", "def"]), "abcdef")

    def test_does_not_insert_paragraph_breaks(self):
        # A "\n\n" here would become a false paragraph boundary that the new
        # chunker splits on, preserving the old cut.
        self.assertNotIn("\n\n", reassemble(["one two thr", "ee four five"]))

    def test_preserves_real_paragraph_structure(self):
        a = "First paragraph.\n\nSecond para"
        b = "Second paragraph continues here."
        out = reassemble([a, b])
        self.assertIn("First paragraph.\n\n", out)
        self.assertEqual(out.count("Second para"), 1)

    def test_ignores_a_trivially_short_coincidental_overlap(self):
        # A single shared character is coincidence, not a real overlap.
        # Dropping it would delete a real character, so both are kept.
        self.assertEqual(reassemble(["abcdefg", "ghijklm"]), "abcdefgghijklm")

    def test_empty_chunks_are_skipped(self):
        # Overlap must clear the floor to be treated as one, so this uses a
        # realistic shared region rather than a few characters.
        self.assertEqual(
            reassemble(["alpha beta gamma delta", "", "beta gamma delta epsilon"]),
            "alpha beta gamma delta epsilon",
        )


if __name__ == "__main__":
    unittest.main()
