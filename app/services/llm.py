from typing import List
from huggingface_hub import InferenceClient
from app.core.config import settings
from app.core.logging_config import logger


class LLMServiceHuggingFace:
    def __init__(self):
        self.model = "HuggingFaceTB/SmolLM3-3B"
        self.client = InferenceClient(
            model=self.model,
            token=settings.HUGGING_FACE_KEY,
        )

    async def generate_response(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str = "helpful and professional",
    ) -> str:
        """
        Generate a response using Hugging Face Inference API.
        """

        try:
            # Build chat messages
            messages = self._build_messages(
                message=message,
                context=context,
                history=history,
                personality=personality,
            )

            logger.info(f"Generated chat messages: {messages}")

            # ✅ Correct HuggingFace call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

            logger.info(f"Raw response from HuggingFace: {response}")

            # ✅ Extract reply safely
            reply = response.choices[0].message["content"]
            logger.info(f"Generated response: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _build_messages(
        self, message: str, context: List[str], history: str, personality: str
    ) -> List[dict]:
        """
        Build OpenAI-style messages with context and history.
        """

        # Put context directly into system prompt to ensure model sees it
        context_text = (
            "\n".join([f"- {ctx}" for ctx in context[:3]])
            if context
            else "No context provided."
        )

        system_prompt = f"""
You are a {personality} AI assistant.
You must use the CONTEXT below when answering.

CONTEXT:
{context_text}

Guidelines:
- Be concise and helpful
- If the answer is not in the context, say so politely
- Reference the context explicitly when possible
- Keep conversation flow natural
"""

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.append(
                {"role": "system", "content": f"Previous Conversation:\n{history}"}
            )

        messages.append({"role": "user", "content": message})

        return messages
