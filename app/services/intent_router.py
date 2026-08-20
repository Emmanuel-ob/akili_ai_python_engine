"""
Intent classification via Bedrock structured output.

Engine Revamp Phase 3. Replaces two pieces of substring keyword matching:

  Issue 01  Handover fired when a message merely CONTAINED a keyword, and the
            list included "support", "manager", "order" and "contact". So "do
            you support Mastercard?" ejected the customer to a human queue.
            The negation guard compounded it by testing for the substring
            "no", which lives inside "know" and "now", so "I know I want to
            talk to a human" was read as a negation and did NOT hand over.

  Issue 05  SQL-versus-FAQ routing counted substring hits in two hand-written
            lists. "order" matched "in order to". "last" matched "last time".
            "contact" appeared in the FAQ list and the handover list at once.

Both are now one tool-use call returning route, wants_human and confidence
together, so routing and handover can never disagree about the same message.

Design notes inherited from Trivia's voice intent router, which has been in
production long enough to learn them:

  * Some Nova builds reject a forced toolChoice with a ValidationException.
    Force the tool first for clean structured output, and on failure retry
    with auto and parse whatever text comes back.
  * boto3 is synchronous. Callers must run classify() in a threadpool, never
    directly on the event loop.
  * Every failure degrades to a safe default. A classifier outage must never
    start ejecting customers into a human queue.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.core.logging_config import logger

VALID_ROUTES = ("FAQ", "SQL", "HYBRID")

# Returned whenever the classifier cannot be trusted: it is disabled, it
# errored, it timed out, or it answered below the confidence threshold.
# FAQ is the safe route because it works with or without a database, and
# wants_human is False because a wrong handover is worse than a wrong answer.
DEFAULT_INTENT: Dict[str, Any] = {
    "route": "FAQ",
    "wants_human": False,
    "confidence": 0.0,
    "reason": "classifier unavailable",
}

# How many prior turns the classifier sees. Enough to resolve "which ones?"
# against the previous answer, short enough to keep latency and cost down.
HISTORY_TURNS = 6

_INTENT_TOOL_NAME = "classify_message"

INTENT_TOOL: Dict[str, Any] = {
    "toolSpec": {
        "name": _INTENT_TOOL_NAME,
        "description": (
            "Classify one customer message: decide which knowledge source can "
            "answer it, and whether the customer is asking for a human."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": list(VALID_ROUTES),
                        "description": (
                            "FAQ = answerable from documents, policies or general "
                            "knowledge about the business. "
                            "SQL = needs live business records, e.g. counts, totals, "
                            "order status, a specific customer's data. "
                            "HYBRID = genuinely needs both, e.g. 'what is your refund "
                            "policy and where is my refund?'. Prefer FAQ or SQL; only "
                            "choose HYBRID when one source alone leaves the question "
                            "half-answered."
                        ),
                    },
                    "wants_human": {
                        "type": "boolean",
                        "description": (
                            "True ONLY when the customer is asking to be connected to "
                            "a person, or is frustrated enough that a person should "
                            "step in. A question that merely mentions support, an "
                            "account manager, an order or contact details is NOT a "
                            "request for a human. 'Do you support Mastercard?' is "
                            "false. 'Can I speak to someone?' is true."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "0.0 to 1.0. How certain you are of BOTH fields above. "
                            "Be honest: a low score means the system answers safely "
                            "rather than acting on a guess."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "A few words explaining the choice, for logs.",
                    },
                },
                "required": ["route", "wants_human", "confidence"],
            }
        },
    }
}

_SYSTEM_PROMPT = (
    "You classify incoming messages for a business's customer-support chatbot. "
    "You never answer the customer. You only classify, by calling the "
    "classify_message tool exactly once.\n\n"
    "Judge what the customer MEANS, not which words they used. Words like "
    "support, manager, order and contact appear constantly in ordinary product "
    "questions and are not requests for a human."
)


def build_user_text(
    message: str, history: Optional[List[Dict[str, Any]]], has_database: bool
) -> str:
    """Assemble the classifier's input.

    History matters for follow-ups: "which ones?" is only routable if the
    classifier can see that the previous turn returned data.
    """
    lines: List[str] = []

    if has_database:
        lines.append(
            "This business has a live database connected, so questions about "
            "their records can be answered with SQL."
        )
    else:
        lines.append(
            "This business has no database connected. Records cannot be "
            "queried, so SQL is not available."
        )

    recent = (history or [])[-HISTORY_TURNS:]
    if recent:
        lines.append("\nRecent conversation:")
        for turn in recent:
            who = "Customer" if turn.get("role") == "user" else "Bot"
            text = turn.get("message") or turn.get("content") or ""
            if text:
                lines.append(f"{who}: {text}")

    lines.append(f'\nClassify this message:\n"{message}"')
    return "\n".join(lines)


def _loads_lenient(text: str) -> Dict[str, Any]:
    """Parse JSON that may be wrapped in prose or a code fence.

    Used only on the fallback path, where a forced toolChoice was rejected and
    the model replied in text instead of calling the tool.
    """
    if not text:
        return {}
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if not fenced:
        return {}
    try:
        parsed = json.loads(fenced.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def parse_intent(response: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the classification out of a Converse response.

    Prefers the toolUse block. Falls back to a JSON text block, which is what
    arrives when the model declined to call the tool.
    """
    content = (response or {}).get("output", {}).get("message", {}).get("content") or []

    for block in content:
        tool = block.get("toolUse")
        if tool and tool.get("name") == _INTENT_TOOL_NAME:
            return tool.get("input") or {}

    for block in content:
        if block.get("text"):
            parsed = _loads_lenient(block["text"])
            if parsed:
                return parsed

    return {}


def _normalise(
    raw: Dict[str, Any], has_database: bool, threshold: float
) -> Dict[str, Any]:
    """Validate the model's answer and apply the safety rules.

    Three ways a technically-valid response is still not actionable:
      * confidence below the threshold, so we do not act on a guess
      * a route the enum does not contain, i.e. the model invented one
      * SQL or HYBRID when there is no database to query
    """
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < threshold:
        logger.info(
            f"Intent confidence {confidence} below threshold {threshold}; "
            f"using the safe default"
        )
        return {**DEFAULT_INTENT, "confidence": confidence, "reason": "low confidence"}

    route = raw.get("route")
    if route not in VALID_ROUTES:
        logger.warning(f"Intent router returned an unknown route {route!r}")
        route = "FAQ"

    if route in ("SQL", "HYBRID") and not has_database:
        logger.info(f"Route {route} requested but no database is connected; using FAQ")
        route = "FAQ"

    return {
        "route": route,
        "wants_human": bool(raw.get("wants_human", False)),
        "confidence": confidence,
        "reason": str(raw.get("reason", "")),
    }


def classify(
    message: str,
    history: Optional[List[Dict[str, Any]]],
    has_database: bool,
    model: str,
    region: str,
    threshold: float,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Classify one message. Never raises.

    client is injectable so tests never reach AWS. Blocking boto3 call: run
    this in a threadpool, not on the event loop.
    """
    if not (message or "").strip():
        return DEFAULT_INTENT

    try:
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(retries={"max_attempts": 2, "mode": "adaptive"}),
            )

        user_text = build_user_text(message, history, has_database)

        def _call(force_tool: bool) -> Dict[str, Any]:
            tool_config: Dict[str, Any] = {"tools": [INTENT_TOOL]}
            if force_tool:
                tool_config["toolChoice"] = {"tool": {"name": _INTENT_TOOL_NAME}}
            return client.converse(
                modelId=model,
                system=[{"text": _SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_text}]}],
                toolConfig=tool_config,
                # Classification is not creative work.
                inferenceConfig={"maxTokens": 512, "temperature": 0},
            )

        # Force the tool for clean structured output. Some Nova builds reject
        # a forced toolChoice with a ValidationException, so fall back to auto
        # and parse whatever text comes back.
        try:
            response = _call(force_tool=True)
        except Exception as exc:
            logger.debug(f"Forced toolChoice failed ({exc}); retrying with auto")
            response = _call(force_tool=False)

        raw = parse_intent(response)
        if not raw:
            logger.warning("Intent router returned nothing parseable")
            return DEFAULT_INTENT

        result = _normalise(raw, has_database, threshold)
        logger.info(
            f"Intent: route={result['route']} wants_human={result['wants_human']} "
            f"confidence={result['confidence']} ({result['reason']})"
        )
        return result

    except Exception as exc:
        # Hard Rule 8: never let this reach the request path.
        logger.error(f"Intent classification failed: {exc}", exc_info=True)
        return DEFAULT_INTENT
