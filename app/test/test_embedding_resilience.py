"""
Tests for embedding client resilience.

Pure helpers with no SDK imports, so these run with nothing installed:
    python -m unittest discover -s app/test -p "test_*.py" -v
"""

import unittest

from app.services.embedding_resilience import (
    batch_texts,
    cap_input,
    is_context_length_error,
    with_retry,
)


class TestCapInput(unittest.TestCase):
    def test_long_text_is_truncated(self):
        self.assertEqual(len(cap_input("X" * 9000, 6000)), 6000)

    def test_short_text_is_untouched(self):
        self.assertEqual(cap_input("hello", 6000), "hello")

    def test_empty_text_is_safe(self):
        self.assertEqual(cap_input("", 6000), "")


class TestBatchTexts(unittest.TestCase):
    def test_splits_into_bounded_groups(self):
        batches = list(batch_texts([f"t{i}" for i in range(70)], 32))
        self.assertEqual([len(b) for b in batches], [32, 32, 6])

    def test_empty_input_yields_no_batches(self):
        self.assertEqual(list(batch_texts([], 32)), [])

    def test_batch_larger_than_input_yields_one_batch(self):
        self.assertEqual(len(list(batch_texts(["a", "b"], 32))), 1)

    def test_preserves_order_across_batches(self):
        items = [f"t{i}" for i in range(10)]
        flat = [x for b in batch_texts(items, 3) for x in b]
        self.assertEqual(flat, items)


class TestIsContextLengthError(unittest.TestCase):
    def test_recognises_a_context_length_rejection(self):
        self.assertTrue(
            is_context_length_error(Exception("input length exceeds the context length"))
        )

    def test_recognises_a_token_limit_phrasing(self):
        self.assertTrue(is_context_length_error(Exception("400 input token count exceeds")))

    def test_does_not_flag_a_transient_error(self):
        self.assertFalse(is_context_length_error(Exception("503 Service Unavailable")))


class TestWithRetry(unittest.TestCase):
    def test_returns_on_first_success(self):
        calls = []

        def ok(text):
            calls.append(text)
            return [0.1, 0.2]

        self.assertEqual(
            with_retry(ok, "hello", retries=3, sleep=lambda s: None), [0.1, 0.2]
        )
        self.assertEqual(len(calls), 1)

    def test_retries_a_transient_failure(self):
        attempts = []

        def flaky(text):
            attempts.append(text)
            if len(attempts) < 3:
                raise Exception("503 Service Unavailable")
            return [0.1]

        self.assertEqual(with_retry(flaky, "x", retries=3, sleep=lambda s: None), [0.1])
        self.assertEqual(len(attempts), 3)

    def test_backoff_is_exponential(self):
        waits = []

        def always_503(text):
            raise Exception("503")

        with self.assertRaises(Exception):
            with_retry(always_503, "x", retries=4, sleep=waits.append)
        self.assertEqual(waits, [1, 2, 4])

    def test_halves_the_input_on_a_context_length_error(self):
        seen = []

        def too_long(text):
            seen.append(len(text))
            if len(text) > 1000:
                raise Exception("input length exceeds the context length")
            return [0.1]

        with_retry(too_long, "X" * 4000, retries=6, sleep=lambda s: None)
        # Re-sending identical over-length text fails identically forever, so
        # each retry must shrink it.
        self.assertTrue(all(b < a for a, b in zip(seen, seen[1:])))
        self.assertLessEqual(seen[-1], 1000)

    def test_context_length_retry_does_not_sleep(self):
        # Shrinking is deterministic, so there is nothing to wait for.
        waits = []

        def too_long(text):
            if len(text) > 500:
                raise Exception("exceeds the context length")
            return [0.1]

        with_retry(too_long, "X" * 4000, retries=6, sleep=waits.append)
        self.assertEqual(waits, [])

    def test_stops_shrinking_at_a_floor(self):
        # A text that fails no matter how small must not loop forever.
        def always_too_long(text):
            raise Exception("exceeds the context length")

        with self.assertRaises(Exception):
            with_retry(always_too_long, "X" * 4000, retries=6, sleep=lambda s: None)

    def test_a_batch_payload_is_never_sliced(self):
        # The bulk path passes a LIST of texts. Halving a list on an
        # over-length error would silently drop texts and return fewer
        # embeddings than the caller passed in, which is data loss rather than
        # a failure. Only a string payload may be shrunk.
        big_batch = [f"t{i}" for i in range(600)]
        seen = []

        def too_long(payload):
            seen.append(len(payload))
            raise Exception("exceeds the context length")

        with self.assertRaises(Exception):
            with_retry(too_long, big_batch, retries=3, sleep=lambda s: None)
        self.assertEqual(set(seen), {600})

    def test_raises_after_exhausting_retries(self):
        def always_fails(text):
            raise Exception("permanent")

        with self.assertRaises(Exception):
            with_retry(always_fails, "x", retries=2, sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
