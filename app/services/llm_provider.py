"""
Provider-switched text generation.

Engine Revamp Phase 2. Generation moves to Bedrock Nova Lite; Gemini stays
reachable by env var as the rollback path.

Two rules make this module testable on a machine with no engine dependencies
installed, which is why the logic lives here rather than in llm.py:

  1. SDK imports are LAZY, inside the functions that need them. llm.py imports
     google.genai and groq at module scope and constructs a Groq client at
     import time, so it cannot be imported at all without those packages.
  2. Every network call accepts an injectable client. Tests pass a fake.

Settings are passed in by callers rather than read here, so this module has no
import-time dependency on configuration either.
"""

from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_config import logger


def to_bedrock_messages(
    messages: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Convert the internal message format to Bedrock Converse shape.

    Internal format is what LLMService._build_messages already produces:
        [{"role": "system"|"user"|"assistant", "content": str}, ...]

    Bedrock keeps system prompts OUT of the message list, in their own
    top-level block, and wraps each turn's text in a content array. Multiple
    system messages are joined rather than dropped.

    Blank turns are removed: Bedrock rejects an empty content string with a
    validation error, which would fail the whole request.
    """
    system_texts: List[str] = []
    turns: List[Dict[str, Any]] = []

    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        role = msg.get("role")
        if role == "system":
            system_texts.append(content)
        else:
            turns.append({"role": role, "content": [{"text": content}]})

    system_blocks = [{"text": "\n\n".join(system_texts)}] if system_texts else []
    return system_blocks, turns


def generate_bedrock(
    messages: List[Dict[str, str]],
    model: str,
    region: str,
    temperature: float,
    max_tokens: int = 2048,
    client: Optional[Any] = None,
) -> str:
    """Generate text via the Bedrock Converse API.

    client is injectable so tests never reach AWS. In production it is None
    and a boto3 client is built here, which is also where boto3 is imported.

    Raises on failure. Callers own the fallback decision, because only they
    know whether another provider is available.
    """
    if client is None:
        import boto3
        from botocore.config import Config

        # Adaptive retry: a busy widget can produce concurrent calls, and
        # legacy retry mode surfaces ThrottlingException as a hard error.
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )

    system_blocks, turns = to_bedrock_messages(messages)

    kwargs: Dict[str, Any] = {
        "modelId": model,
        "messages": turns,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    # Bedrock rejects system=[]; the key must be absent when there is none.
    if system_blocks:
        kwargs["system"] = system_blocks

    response = client.converse(**kwargs)

    if response.get("stopReason") == "max_tokens":
        logger.warning(
            f"Bedrock response hit the {max_tokens} token cap and was truncated"
        )

    blocks = response.get("output", {}).get("message", {}).get("content", [])
    if not blocks:
        logger.warning("Bedrock returned no content blocks")
        return ""

    return (blocks[0].get("text") or "").strip()


def choose_provider(configured: str, gemini_rate_limited: bool) -> str:
    """Pick the provider for one generation call.

    configured is settings.LLM_PROVIDER. The rate-limit flag only applies to
    the Gemini path: Bedrock has its own throttling, handled by boto3's
    adaptive retry, and is not governed by the process-wide Gemini counter.

    An unrecognised value degrades to the Gemini path rather than raising, so
    a typo in an env var cannot take generation down.
    """
    provider = (configured or "").strip().lower()

    if provider == "bedrock":
        return "bedrock"

    if provider != "gemini":
        logger.warning(
            f"Unknown LLM_PROVIDER {configured!r}; falling back to the gemini path"
        )

    return "groq" if gemini_rate_limited else "gemini"
