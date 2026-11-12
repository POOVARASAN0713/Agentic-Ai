import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv()


def _get_setting(name: str, default: str = "") -> str:
    """Read a setting from the environment, then from Streamlit secrets."""
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
    except FileNotFoundError:
        secret_value = None

    return str(secret_value) if secret_value else default


class AppConfig:
    """
    Centralized configuration management for the AI Customer Support System.
    Loads settings from environment variables or applies sensible defaults.
    """

    def __init__(self):
        # API Keys
        self.GROQ_API_KEY: str = _get_setting("GROQ_API_KEY")
        self.GEMINI_API_KEY: str = _get_setting("GEMINI_API_KEY")

        # LLM Models
        configured_groq_model = _get_setting("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.GROQ_MODEL: str = {
            "llama-3.1-8b-instant": "llama-3.3-70b-versatile",
        }.get(configured_groq_model, configured_groq_model)
        self.GEMINI_MODEL: str = _get_setting("GEMINI_MODEL", "gemini-2.0-flash")

        # Embedding Configuration
        self.EMBEDDING_MODEL_NAME: str = _get_setting(
            "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
        )

        # Retrieval & Confidence Thresholds
        self.FAQ_THRESHOLD: float = float(_get_setting("FAQ_THRESHOLD", "0.75"))
        self.MULTILINGUAL_FAQ_THRESHOLD: float = float(
            _get_setting("MULTILINGUAL_FAQ_THRESHOLD", "0.55")
        )
        self.RAG_THRESHOLD: float = float(_get_setting("RAG_THRESHOLD", "0.60"))
        self.MULTILINGUAL_RAG_THRESHOLD: float = float(
            _get_setting("MULTILINGUAL_RAG_THRESHOLD", "0.55")
        )
        self.TOP_K: int = int(_get_setting("TOP_K", "4"))

        # Vector Database Path
        self.CHROMA_PERSIST_DIR: str = str(BASE_DIR / "vectorstore" / "chroma")

        # Application Mode
        self.DEBUG_MODE: bool = _get_setting("DEBUG_MODE", "true").lower() in ("true", "1", "yes")

    def validate_api_keys(self) -> dict[str, bool]:
        """
        Check which LLM provider API keys are configured.
        """
        return {
            "groq_configured": bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "your_groq_api_key_here"),
            "gemini_configured": bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY != "your_gemini_api_key_here"),
        }


# Global configuration instance
config = AppConfig()
