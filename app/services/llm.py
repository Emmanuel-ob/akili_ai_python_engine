from typing import List
from huggingface_hub import InferenceClient
from app.core.config import settings
from app.core.logging_config import logger


class LLMServiceHuggingFace:
    def __init__(self):
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            token=settings.HUGGING_FACE_KEY,  # correct param
        )

    async def generate_response(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str = "helpful and professional",
    ) -> str:
    

        try:
            # Build chat messages
            messages = self._build_messages(
                message=message,
                context=context,
                history=history,
                personality=personality,
            )

            logger.info(f"Generated chat messages: {messages}")

            # Call chat_completion
            response = self.client.chat_completion(
                messages=messages, max_tokens=300, temperature=0.7
            )

            # Extract content safely
            reply = response.choices[0].message["content"]
            logger.info(f"Generated response: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _build_messages(
        self, message: str, context: List[str], history: str, personality: str
    ) -> List[dict]:
       

        system_prompt = f"""
                            You are a {personality} AI assistant. 
                            You help users by answering their questions based on the provided context.
                            - Be concise and helpful
                            - If you don't know, say so politely
                            - Reference context when relevant
                            - Keep conversation flow            
                        """

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            context_text = "\n".join([f"- {ctx[:300]}..." for ctx in context[:3]])
            messages.append(
                {"role": "system", "content": f"Context Information:\n{context_text}"}
            )

        if history:
            messages.append(
                {"role": "system", "content": f"Previous Conversation:\n{history}"}
            )

        messages.append({"role": "user", "content": message})

        return messages
