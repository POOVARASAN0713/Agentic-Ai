"""
LLM Manager Module.
Implements centralized LLM invocation with automated fallback mechanism:
Primary Provider: Groq (using Llama 3.3 70B)
Fallback Provider: Google Gemini (using Gemini 2.0 Flash)
"""

from typing import Optional, Dict, Any
from utils.config import config
from utils.logging_utils import get_logger

logger = get_logger("LLMManager")


class LLMManager:
    """
    Manages LLM instances and handles fallback logic gracefully.
    """

    def __init__(self):
        self.groq_key = config.GROQ_API_KEY
        self.gemini_key = config.GEMINI_API_KEY
        self.groq_model_name = config.GROQ_MODEL
        self.gemini_model_name = config.GEMINI_MODEL

    def _get_groq_llm(self, temperature: float = 0.0):
        """Initialize Groq Chat LLM instance."""
        if not config.validate_api_keys()["groq_configured"]:
            raise ValueError("GROQ_API_KEY is missing or unconfigured.")
        
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=self.groq_key,
            model_name=self.groq_model_name,
            temperature=temperature,
        )

    def _get_gemini_llm(self, temperature: float = 0.0):
        """Initialize Google Gemini Chat LLM instance."""
        if not config.validate_api_keys()["gemini_configured"]:
            raise ValueError("GEMINI_API_KEY is missing or unconfigured.")
        
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=self.gemini_key,
            model=self.gemini_model_name,
            temperature=temperature,
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize string and structured provider response content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Invokes the primary LLM (Groq) and falls back to Gemini if Groq fails.

        Args:
            prompt: The user query or prompt.
            system_prompt: Optional instruction prompt.
            temperature: Sampling temperature (0.0 for deterministic answers).

        Returns:
            Dict containing response text, provider used, and status.
        """
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("user", prompt))

        # 1. Try Primary Provider (Groq)
        try:
            logger.info(f"Attempting invocation via Primary Provider (Groq: {self.groq_model_name})...")
            groq_llm = self._get_groq_llm(temperature=temperature)
            response = groq_llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else response
            content = self._content_to_text(raw_content)

            logger.info("Successfully received response from Groq.")
            return {
                "content": content,
                "provider": "Groq",
                "model": self.groq_model_name,
                "status": "success",
                "fallback_used": False,
            }

        except Exception as groq_err:
            logger.warning(f"Groq API call failed: {str(groq_err)}. Initiating fallback to Gemini...")

            # 2. Try Fallback Provider (Gemini)
            try:
                gemini_llm = self._get_gemini_llm(temperature=temperature)
                response = gemini_llm.invoke(messages)
                raw_content = response.content if hasattr(response, "content") else response
                content = self._content_to_text(raw_content)

                logger.info("Successfully received response from Gemini (Fallback).")
                return {
                    "content": content,
                    "provider": "Gemini",
                    "model": self.gemini_model_name,
                    "status": "success",
                    "fallback_used": True,
                }

            except Exception as gemini_err:
                logger.error(f"Gemini Fallback API call failed: {str(gemini_err)}")

                # 3. Both Failed
                return {
                    "content": (
                        "I am temporarily unable to process your request due to service unavailability. "
                        "Please check back shortly or reach out to our support team."
                    ),
                    "provider": None,
                    "model": None,
                    "status": "error",
                    "fallback_used": True,
                    "error_details": f"Groq Error: {str(groq_err)} | Gemini Error: {str(gemini_err)}",
                }


# Global LLM Manager Singleton
llm_manager = LLMManager()
