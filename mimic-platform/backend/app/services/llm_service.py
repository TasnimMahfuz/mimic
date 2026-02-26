import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


class LLMService:
    """
    Service to handle Groq API calls for text generation.
    Uses cloud-based Groq API - no local models.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        self.client = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=512,
            timeout=30
        )

    def generate_response(self, context: str, query: str) -> str:
        # UPDATED: Cleaner separation of instructions and data
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant for the MIMIC platform. "
                "Use the provided context to answer the question. "
                "If the context is insufficient, explain that you don't have that information."
            )),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
        ]
        
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            # INSERTED: Catch API errors (Rate limits, connection issues)
            print(f"Groq API Error: {e}") 
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The AI service is temporarily busy. Please try again."
            )

    def generate_response_with_fallback(self, context: str, query: str) -> str:
        """
        Generate response with fallback for API errors.
        """
        try:
            return self.generate_response(context, query)
        except Exception as e:
            return f"Error generating response: {str(e)}"