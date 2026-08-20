"""
Tests for retrieval gating and ordering.

Run from the engine root with no dependencies installed:

    python -m unittest discover -s app/test -p "test_*.py" -v

These cover the two Phase 1 defects from the engine revamp charter:

  Issue 02  the confidence floor was hardcoded at 0.20 in two call sites while
            settings.MIN_CONFIDENCE_THRESHOLD (0.5) was declared and never read.

  Issue 03  results were re-sorted by admin priority BEFORE the confidence gate
            ran, so a high-priority irrelevant chunk could displace the best
            match and become both the thing that opened the gate and the answer.

test_issue_03_reproduces_old_behaviour below encodes the OLD algorithm and
asserts it picks the wrong chunk, so the regression is documented rather than
merely fixed.
"""

import unittest

from app.services.retrieval_gate import gate_and_order, passes_threshold


class Hit:
    """Minimal stand-in for a vector store hit.

    The real object comes from the vector store client and carries a .score
    plus a .payload dict. Only those two are used by the gate, so the tests
    do not need the client installed.
    """

    def __init__(self, doc_id, score, priority=5):
        self.score = score
        self.payload = {
            "doc_id": doc_id,
            "text": f"text of {doc_id}",
            "metadata": {"faq_priority": priority},
        }

    def __repr__(self):
        return f"<Hit {self.payload['doc_id']} score={self.score} prio={self.payload['metadata']['faq_priority']}>"

    @property
    def doc_id(self):
        return self.payload["doc_id"]


class TestPassesThreshold(unittest.TestCase):
    def test_score_above_threshold_passes(self):
        self.assertTrue(passes_threshold(0.61, 0.5))

    def test_score_exactly_at_threshold_passes(self):
        # Boundary is inclusive: a chunk scoring exactly the floor is relevant.
        self.assertTrue(passes_threshold(0.5, 0.5))

    def test_score_below_threshold_fails(self):
        self.assertFalse(passes_threshold(0.49, 0.5))

    def test_the_old_hardcoded_floor_is_no_longer_special(self):
        # 0.20 was the hardcoded value. At a 0.5 floor it must now fail.
        self.assertFalse(passes_threshold(0.20, 0.5))


class TestGateAndOrder(unittest.TestCase):
    def test_returns_empty_when_nothing_clears_the_floor(self):
        hits = [Hit("a", 0.31), Hit("b", 0.22), Hit("c", 0.10)]
        self.assertEqual(gate_and_order(hits, threshold=0.5), [])

    def test_drops_only_the_chunks_below_the_floor(self):
        hits = [Hit("keep", 0.72), Hit("drop", 0.19), Hit("keep2", 0.55)]
        got = [h.doc_id for h in gate_and_order(hits, threshold=0.5)]
        self.assertEqual(sorted(got), ["keep", "keep2"])

    def test_priority_orders_only_among_survivors(self):
        # The high-priority chunk is irrelevant (0.21) and must not survive,
        # even though its priority is the highest in the set.
        hits = [
            Hit("irrelevant_but_top_priority", 0.21, priority=10),
            Hit("relevant_low_priority", 0.81, priority=1),
        ]
        got = [h.doc_id for h in gate_and_order(hits, threshold=0.5)]
        self.assertEqual(got, ["relevant_low_priority"])

    def test_priority_wins_between_two_relevant_chunks(self):
        # Both clear the floor, so the admin's priority is respected.
        hits = [
            Hit("lower_priority", 0.88, priority=2),
            Hit("higher_priority", 0.62, priority=9),
        ]
        got = [h.doc_id for h in gate_and_order(hits, threshold=0.5)]
        self.assertEqual(got, ["higher_priority", "lower_priority"])

    def test_score_breaks_ties_within_equal_priority(self):
        hits = [
            Hit("weaker", 0.55, priority=5),
            Hit("stronger", 0.91, priority=5),
        ]
        got = [h.doc_id for h in gate_and_order(hits, threshold=0.5)]
        self.assertEqual(got, ["stronger", "weaker"])

    def test_missing_priority_metadata_does_not_crash(self):
        bare = Hit("bare", 0.77)
        del bare.payload["metadata"]
        got = [h.doc_id for h in gate_and_order([bare], threshold=0.5)]
        self.assertEqual(got, ["bare"])

    def test_empty_input(self):
        self.assertEqual(gate_and_order([], threshold=0.5), [])

    def test_limit_caps_the_result(self):
        hits = [Hit(f"h{i}", 0.9 - i * 0.01, priority=5) for i in range(10)]
        self.assertEqual(len(gate_and_order(hits, threshold=0.5, limit=3)), 3)


class TestIssue03Regression(unittest.TestCase):
    """Documents the defect being fixed, so it cannot silently return."""

    HITS = [
        # Genuinely the best match, but the admin left it at default priority.
        Hit("correct_answer", 0.84, priority=5),
        # Irrelevant, but an admin pinned it to the top priority.
        Hit("pinned_but_irrelevant", 0.23, priority=10),
    ]

    @staticmethod
    def _old_algorithm(results, floor=0.20):
        """The behaviour that shipped, reproduced from chat.py:662-670.

        Sort every hit by priority first, then test only results[0] against
        the floor and return it.
        """
        ordered = sorted(
            results,
            key=lambda x: x.payload.get("metadata", {}).get("faq_priority", 5),
            reverse=True,
        )
        top = ordered[0]
        if top.score < floor:
            return []
        return ordered[:3]

    def test_old_behaviour_answered_from_the_wrong_chunk(self):
        got = [h.doc_id for h in self._old_algorithm(self.HITS)]
        # The pinned, irrelevant chunk was both the gate and the first answer.
        self.assertEqual(got[0], "pinned_but_irrelevant")

    def test_new_behaviour_answers_from_the_relevant_chunk(self):
        got = [h.doc_id for h in gate_and_order(self.HITS, threshold=0.5)]
        self.assertEqual(got, ["correct_answer"])

    def test_new_behaviour_excludes_the_irrelevant_chunk_entirely(self):
        got = [h.doc_id for h in gate_and_order(self.HITS, threshold=0.5)]
        self.assertNotIn("pinned_but_irrelevant", got)


if __name__ == "__main__":
    unittest.main()
