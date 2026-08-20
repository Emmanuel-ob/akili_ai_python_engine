"""
Tests for the Bedrock intent router.

Run with no engine dependencies installed:
    python -m unittest discover -s app/test -p "test_*.py" -v

The boto3 client is injected, so these never reach AWS.
"""

import unittest

from app.services.intent_router import (
    DEFAULT_INTENT,
    INTENT_TOOL,
    build_user_text,
    classify,
    parse_intent,
)


def _tool_response(**fields):
    """A Converse response carrying a toolUse block."""
    return {
        "output": {
            "message": {
                "content": [{"toolUse": {"name": "classify_message", "input": fields}}]
            }
        }
    }


class FakeClient:
    def __init__(self, response=None, fail_on_forced=False):
        self._response = response or _tool_response(
            route="FAQ", wants_human=False, confidence=0.9, reason="general question"
        )
        self.fail_on_forced = fail_on_forced
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        forced = "toolChoice" in (kwargs.get("toolConfig") or {})
        if forced and self.fail_on_forced:
            raise Exception("ValidationException: toolChoice not supported")
        return self._response


class TestToolSchema(unittest.TestCase):
    def test_tool_is_named_and_described(self):
        spec = INTENT_TOOL["toolSpec"]
        self.assertEqual(spec["name"], "classify_message")
        self.assertTrue(spec["description"])

    def test_route_enum_covers_exactly_the_three_routes(self):
        props = INTENT_TOOL["toolSpec"]["inputSchema"]["json"]["properties"]
        self.assertEqual(set(props["route"]["enum"]), {"FAQ", "SQL", "HYBRID"})

    def test_required_fields_are_declared(self):
        schema = INTENT_TOOL["toolSpec"]["inputSchema"]["json"]
        self.assertEqual(set(schema["required"]), {"route", "wants_human", "confidence"})


class TestParseIntent(unittest.TestCase):
    def test_extracts_the_tool_input(self):
        got = parse_intent(
            _tool_response(route="SQL", wants_human=False, confidence=0.8, reason="r")
        )
        self.assertEqual(got["route"], "SQL")
        self.assertEqual(got["confidence"], 0.8)

    def test_falls_back_to_a_json_text_block(self):
        # When toolChoice is rejected the model answers in text instead.
        resp = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '{"route":"HYBRID","wants_human":false,"confidence":0.75}'
                        }
                    ]
                }
            }
        }
        self.assertEqual(parse_intent(resp)["route"], "HYBRID")

    def test_tolerates_a_fenced_json_text_block(self):
        resp = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '```json\n{"route":"FAQ","wants_human":true,"confidence":0.9}\n```'
                        }
                    ]
                }
            }
        }
        self.assertTrue(parse_intent(resp)["wants_human"])

    def test_empty_content_returns_empty_dict(self):
        self.assertEqual(parse_intent({"output": {"message": {"content": []}}}), {})

    def test_unparseable_text_returns_empty_dict(self):
        resp = {"output": {"message": {"content": [{"text": "I think it is FAQ"}]}}}
        self.assertEqual(parse_intent(resp), {})


class TestClassify(unittest.TestCase):
    def test_returns_the_models_route(self):
        fake = FakeClient(
            _tool_response(
                route="SQL", wants_human=False, confidence=0.95, reason="count"
            )
        )
        got = classify("how many orders?", [], True, "m", "r", 0.7, client=fake)
        self.assertEqual(got["route"], "SQL")

    def test_low_confidence_falls_back_to_faq_and_no_handover(self):
        fake = FakeClient(
            _tool_response(
                route="SQL", wants_human=True, confidence=0.3, reason="unsure"
            )
        )
        got = classify("hmm", [], True, "m", "r", 0.7, client=fake)
        self.assertEqual(got["route"], "FAQ")
        self.assertFalse(got["wants_human"])

    def test_sql_route_downgraded_to_faq_without_a_database(self):
        fake = FakeClient(
            _tool_response(route="SQL", wants_human=False, confidence=0.9, reason="x")
        )
        got = classify("how many orders?", [], False, "m", "r", 0.7, client=fake)
        self.assertEqual(got["route"], "FAQ")

    def test_provider_failure_degrades_to_the_safe_default(self):
        class Boom:
            def converse(self, **kw):
                raise Exception("network down")

        got = classify("anything", [], True, "m", "r", 0.7, client=Boom())
        self.assertEqual(got, DEFAULT_INTENT)

    def test_safe_default_never_hands_over(self):
        # A classifier outage must not start ejecting customers to a queue.
        self.assertFalse(DEFAULT_INTENT["wants_human"])
        self.assertEqual(DEFAULT_INTENT["route"], "FAQ")

    def test_retries_with_auto_when_forced_tool_choice_is_rejected(self):
        fake = FakeClient(fail_on_forced=True)
        got = classify("hello", [], False, "m", "r", 0.7, client=fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("toolChoice", fake.calls[0]["toolConfig"])
        self.assertNotIn("toolChoice", fake.calls[1]["toolConfig"])
        self.assertEqual(got["route"], "FAQ")

    def test_temperature_is_zero(self):
        # Classification is not creative work.
        fake = FakeClient()
        classify("hi", [], False, "m", "r", 0.7, client=fake)
        self.assertEqual(fake.calls[0]["inferenceConfig"]["temperature"], 0)

    def test_unknown_route_value_degrades_to_faq(self):
        fake = FakeClient(
            _tool_response(
                route="NONSENSE", wants_human=False, confidence=0.9, reason=""
            )
        )
        got = classify("hi", [], True, "m", "r", 0.7, client=fake)
        self.assertEqual(got["route"], "FAQ")

    def test_missing_confidence_is_treated_as_zero(self):
        fake = FakeClient(_tool_response(route="SQL", wants_human=True))
        got = classify("hi", [], True, "m", "r", 0.7, client=fake)
        self.assertEqual(got["route"], "FAQ")
        self.assertFalse(got["wants_human"])


class TestBuildUserText(unittest.TestCase):
    def test_includes_the_message(self):
        self.assertIn("cancel my order", build_user_text("cancel my order", [], True))

    def test_states_whether_a_database_is_connected(self):
        self.assertIn("no database", build_user_text("hi", [], False).lower())

    def test_includes_recent_history(self):
        history = [
            {"role": "user", "message": "how many orders?"},
            {"role": "assistant", "message": "You have 12."},
        ]
        text = build_user_text("which ones?", history, True)
        self.assertIn("how many orders?", text)

    def test_history_is_capped(self):
        history = [{"role": "user", "message": f"m{i}"} for i in range(50)]
        text = build_user_text("x", history, True)
        self.assertNotIn("m0", text)


class TestIssue01Regression(unittest.TestCase):
    """The keyword handover bugs this phase deletes."""

    def test_support_as_a_verb_no_longer_forces_handover(self):
        # "do you support Mastercard?" hit the `support` keyword and ejected
        # the customer to a human queue.
        fake = FakeClient(
            _tool_response(
                route="FAQ",
                wants_human=False,
                confidence=0.93,
                reason="product question",
            )
        )
        got = classify("do you support Mastercard?", [], False, "m", "r", 0.7, client=fake)
        self.assertFalse(got["wants_human"])

    def test_know_no_longer_reads_as_a_negation(self):
        # The old guard tested the substring "no", which is inside "know", so
        # a genuine request was read as a negation and did NOT hand over.
        fake = FakeClient(
            _tool_response(
                route="FAQ",
                wants_human=True,
                confidence=0.96,
                reason="explicit request",
            )
        )
        got = classify(
            "I know I want to talk to a human", [], False, "m", "r", 0.7, client=fake
        )
        self.assertTrue(got["wants_human"])


if __name__ == "__main__":
    unittest.main()
