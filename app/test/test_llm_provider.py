"""
Tests for provider-switched text generation.

Run with no engine dependencies installed:
    python -m unittest discover -s app/test -p "test_*.py" -v

SDK imports in llm_provider are lazy (inside functions), and every network
call takes an injectable client, so these tests never touch AWS.
"""

import unittest

from app.services.llm_provider import (
    choose_provider,
    generate_bedrock,
    to_bedrock_messages,
)


class FakeBedrockClient:
    """Records the call and returns a canned Converse response."""

    def __init__(self, text="canned reply", stop_reason="end_turn"):
        self._text = text
        self._stop = stop_reason
        self.last_kwargs = None
        self.call_count = 0

    def converse(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        return {
            "stopReason": self._stop,
            "output": {"message": {"content": [{"text": self._text}]}},
        }


class TestToBedrockMessages(unittest.TestCase):
    def test_system_messages_are_separated(self):
        system, msgs = to_bedrock_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        )
        self.assertEqual(system, [{"text": "You are helpful."}])
        self.assertEqual(msgs, [{"role": "user", "content": [{"text": "Hello"}]}])

    def test_multiple_system_messages_are_joined(self):
        system, _ = to_bedrock_messages(
            [
                {"role": "system", "content": "Rule one."},
                {"role": "system", "content": "Rule two."},
                {"role": "user", "content": "Hi"},
            ]
        )
        self.assertEqual(system, [{"text": "Rule one.\n\nRule two."}])

    def test_assistant_turns_are_preserved(self):
        _, msgs = to_bedrock_messages(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
            ]
        )
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])

    def test_no_system_message_yields_empty_system_blocks(self):
        system, msgs = to_bedrock_messages([{"role": "user", "content": "Hi"}])
        self.assertEqual(system, [])
        self.assertEqual(len(msgs), 1)

    def test_empty_content_messages_are_dropped(self):
        # A blank turn is rejected by Bedrock with a validation error.
        _, msgs = to_bedrock_messages(
            [
                {"role": "user", "content": "Real"},
                {"role": "assistant", "content": "   "},
            ]
        )
        self.assertEqual(len(msgs), 1)


class TestGenerateBedrock(unittest.TestCase):
    def test_returns_the_model_text(self):
        fake = FakeBedrockClient(text="Spring collection is live")
        out = generate_bedrock(
            [{"role": "user", "content": "Hi"}],
            model="amazon.nova-lite-v1:0",
            region="us-east-1",
            temperature=0.1,
            client=fake,
        )
        self.assertEqual(out, "Spring collection is live")

    def test_passes_model_and_temperature_through(self):
        fake = FakeBedrockClient()
        generate_bedrock(
            [{"role": "user", "content": "Hi"}],
            model="amazon.nova-lite-v1:0",
            region="us-east-1",
            temperature=0.1,
            client=fake,
        )
        self.assertEqual(fake.last_kwargs["modelId"], "amazon.nova-lite-v1:0")
        self.assertEqual(fake.last_kwargs["inferenceConfig"]["temperature"], 0.1)

    def test_system_block_omitted_entirely_when_there_is_none(self):
        # Sending system=[] is a validation error; the key must be absent.
        fake = FakeBedrockClient()
        generate_bedrock(
            [{"role": "user", "content": "Hi"}],
            model="m",
            region="r",
            temperature=0.1,
            client=fake,
        )
        self.assertNotIn("system", fake.last_kwargs)

    def test_system_block_present_when_there_is_one(self):
        fake = FakeBedrockClient()
        generate_bedrock(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
            ],
            model="m",
            region="r",
            temperature=0.1,
            client=fake,
        )
        self.assertEqual(fake.last_kwargs["system"], [{"text": "Be brief."}])

    def test_empty_response_content_returns_empty_string(self):
        fake = FakeBedrockClient()
        fake.converse = lambda **kw: {
            "stopReason": "end_turn",
            "output": {"message": {"content": []}},
        }
        out = generate_bedrock(
            [{"role": "user", "content": "Hi"}],
            model="m",
            region="r",
            temperature=0.1,
            client=fake,
        )
        self.assertEqual(out, "")

    def test_result_is_stripped(self):
        fake = FakeBedrockClient(text="  padded  ")
        out = generate_bedrock(
            [{"role": "user", "content": "Hi"}],
            model="m",
            region="r",
            temperature=0.1,
            client=fake,
        )
        self.assertEqual(out, "padded")


class TestChooseProvider(unittest.TestCase):
    def test_bedrock_when_configured(self):
        self.assertEqual(choose_provider("bedrock", gemini_rate_limited=False), "bedrock")

    def test_bedrock_is_unaffected_by_the_gemini_rate_limiter(self):
        # The Gemini quota is irrelevant once generation is on Bedrock. This
        # is the bug the cutover retires: a shared process-wide counter keyed
        # on the bare string "gemini" pushed every business onto the fallback.
        self.assertEqual(choose_provider("bedrock", gemini_rate_limited=True), "bedrock")

    def test_gemini_when_configured_and_not_limited(self):
        self.assertEqual(choose_provider("gemini", gemini_rate_limited=False), "gemini")

    def test_groq_when_gemini_configured_but_limited(self):
        self.assertEqual(choose_provider("gemini", gemini_rate_limited=True), "groq")

    def test_unknown_provider_falls_back_to_gemini_path(self):
        # A typo in LLM_PROVIDER must not take the bot down.
        self.assertEqual(choose_provider("nonsense", gemini_rate_limited=False), "gemini")

    def test_provider_name_is_case_insensitive_and_trimmed(self):
        self.assertEqual(choose_provider("  BEDROCK ", gemini_rate_limited=False), "bedrock")


if __name__ == "__main__":
    unittest.main()
