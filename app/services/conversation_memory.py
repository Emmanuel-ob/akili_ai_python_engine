from typing import Dict, List, Any, Optional
from datetime import datetime
from app.core.logging_config import logger


class ConversationMemory:
    """Maintains context across conversation turns"""

    def __init__(self):
        self.context_stack: List[Dict] = []
        self.entities_mentioned: Dict[str, Any] = {}
        self.current_topic: Optional[str] = None
        self.pending_actions: List[Dict] = []
        self.max_context_turns = 5

    def update_context(self, user_message: str, intent: Dict[str, Any], response: str):
        """Track conversation context"""

        # Extract entities (order IDs, product names, etc.)
        entities = self._extract_entities(user_message, intent)
        self.entities_mentioned.update(entities)

        # Update topic
        if intent.get("intent_category") != "general_chat":
            self.current_topic = intent.get("description", "unknown")

        # Add to context stack
        self.context_stack.append(
            {
                "user_message": user_message,
                "intent": intent,
                "entities": entities,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only recent turns
        if len(self.context_stack) > self.max_context_turns:
            self.context_stack.pop(0)

    def _extract_entities(self, message: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mentioned entities from message"""

        entities = {}

        # Extract order IDs (common patterns)
        import re

        order_match = re.search(r"order\s*#?(\d+)", message, re.IGNORECASE)
        if order_match:
            entities["order_id"] = order_match.group(1)
            entities["last_queried_item"] = {
                "type": "order",
                "id": order_match.group(1),
            }

        # Extract from intent
        if "specific_fields" in intent:
            entities["queried_fields"] = intent["specific_fields"]

        if "target_tables" in intent:
            entities["queried_tables"] = intent["target_tables"]

        return entities

    def resolve_reference(self, message: str) -> Dict[str, Any]:
        """Handle pronouns and references to previous context"""

        message_lower = message.lower()

        # Check if this is a follow-up question
        followup_indicators = [
            "what about",
            "and the",
            "how about",
            "also",
            "too",
            "is it",
            "does it",
            "can you",
            "show me",
        ]

        is_followup = any(
            indicator in message_lower for indicator in followup_indicators
        )

        if is_followup and self.entities_mentioned.get("last_queried_item"):
            return {
                "is_followup": True,
                "referenced_item": self.entities_mentioned["last_queried_item"],
                "current_topic": self.current_topic,
            }

        return {"is_followup": False}

    def get_context_summary(self) -> str:
        """Get summary of recent conversation"""

        if not self.context_stack:
            return ""

        summary_parts = []
        for ctx in self.context_stack[-3:]:  # Last 3 turns
            summary_parts.append(
                f"User asked about: {ctx['intent'].get('description', 'something')}"
            )

        return " | ".join(summary_parts)

    def clear_context(self):
        """Clear conversation memory"""
        self.context_stack.clear()
        self.entities_mentioned.clear()
        self.current_topic = None
        self.pending_actions.clear()


class ConversationSummarizer:
    """Summarizes old messages to manage context window"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def summarize_old_messages(self, messages: List[Dict]) -> str:
        """Compress old conversation into summary"""

        if len(messages) < 10:
            return ""

        old_messages = messages[:-5]  # Everything except last 5

        # Format messages for summarization
        formatted = []
        for msg in old_messages:
            role = msg.get("role", "user")
            content = msg.get("message", "")
            formatted.append(f"{role}: {content}")

        conversation_text = "\n".join(formatted)

        prompt = f"""Summarize this conversation concisely:

{conversation_text}

Focus on:
- What the customer asked about
- What information was provided
- Any unresolved issues
- Key entities mentioned (order IDs, product names, etc.)

Keep summary under 200 words."""

        try:
            summary = await self.llm.generate_response(
                message=prompt,
                context=[],
                history="",
                personality="concise and factual",
            )
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return "Previous conversation history available."

    async def build_context_with_summary(self, all_messages: List[Dict]) -> str:
        """Build context using summary for old messages"""

        if len(all_messages) <= 10:
            # No summarization needed
            return self._format_messages(all_messages)

        summary = await self.summarize_old_messages(all_messages)
        recent = all_messages[-5:]

        context = f"Earlier conversation summary: {summary}\n\n"
        context += "Recent messages:\n"
        context += self._format_messages(recent)

        return context

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages as text"""
        formatted = []
        for msg in messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("message", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
