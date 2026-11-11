"""
FAQ Specialist Agent Module.
Handles fast semantic similarity search against the ChromaDB FAQ collection,
applying strict confidence thresholding to return accurate answers or trigger escalation.
"""

from typing import Dict, Any, Optional
from utils.config import config
from utils.llm import llm_manager
from utils.vectorstore import vector_store
from utils.logging_utils import get_logger

logger = get_logger("FAQAgent")

FAQ_TRANSLATION_PROMPT = """You are a multilingual customer support translator.
Answer the user's question in the exact same language and writing system used by the user.
Translate only the supplied verified FAQ answer. Do not add facts, explanations, or English text.
Return only the final customer-facing answer.

User question:
{query}

Verified FAQ answer:
{answer}
"""


def _needs_translation(query: str) -> bool:
    """Detect non-English scripts and common Latin-script language markers."""
    normalized = query.lower()
    if any(ord(character) > 127 for character in query):
        return True
    language_markers = (
        "qué", "cuál", "cómo", "dónde", "reembolso", "bonjour", "merci",
        "comment", "wie", "warum", "por favor", "नमस्ते", "வணக்கம்",
    )
    return any(marker in normalized for marker in language_markers)


def _answer_in_query_language(query: str, answer: str) -> str:
    """Translate a verified FAQ answer while preserving the user's language."""
    if not _needs_translation(query):
        return answer

    result = llm_manager.invoke(
        prompt=query,
        system_prompt=FAQ_TRANSLATION_PROMPT.format(query=query, answer=answer),
        temperature=0.0,
    )
    if result["status"] == "error":
        logger.warning("Unable to translate FAQ answer into the user's language.")
        return answer
    return result["content"].strip()


class FAQAgent:
    """
    Specialist Agent responsible for handling frequent customer support questions.
    """

    def __init__(self, default_threshold: Optional[float] = None):
        self.threshold = default_threshold if default_threshold is not None else config.FAQ_THRESHOLD

    def process(self, query: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Processes a user query by retrieving the top matching FAQ embedding.

        Args:
            query: User customer support query.
            threshold: Optional custom confidence threshold override.

        Returns:
            Dict containing agent name, answer, confidence score, source metadata, and escalation status.
        """
        active_threshold = threshold if threshold is not None else self.threshold
        if _needs_translation(query or ""):
            active_threshold = min(active_threshold, config.MULTILINGUAL_FAQ_THRESHOLD)
        logger.info(f"Processing query via FAQ Agent (Threshold={active_threshold}): '{query}'")

        if not query or not query.strip():
            return {
                "agent": "faq",
                "answer": "Empty query provided.",
                "confidence": 0.0,
                "source": None,
                "matched_question": None,
                "should_escalate": True,
            }

        # Retrieve top 1 vector match from ChromaDB FAQ collection
        matches = vector_store.search_faqs(query=query.strip(), top_k=1)

        if not matches:
            logger.info("No FAQ matches returned from vector database.")
            return {
                "agent": "faq",
                "answer": "No reliable FAQ match found.",
                "confidence": 0.0,
                "source": None,
                "matched_question": None,
                "should_escalate": True,
            }

        top_match = matches[0]
        confidence = top_match["similarity_score"]
        matched_question = top_match["question"]
        answer = top_match["answer"]
        faq_id = top_match["id"]
        category = top_match["category"]

        logger.info(f"Top FAQ Candidate: ID={faq_id} [{category}] Q='{matched_question}' | Score={confidence}")

        # Threshold Decision Evaluation
        if confidence >= active_threshold:
            logger.info(f"FAQ Match CONFIRMED (Score {confidence} >= Threshold {active_threshold}).")
            return {
                "agent": "faq",
                "answer": _answer_in_query_language(query.strip(), answer),
                "confidence": confidence,
                "source": f"FAQ #{faq_id} ({category})",
                "matched_question": matched_question,
                "should_escalate": False,
            }
        else:
            logger.info(f"FAQ Match REJECTED (Score {confidence} < Threshold {active_threshold}). Triggering escalation recommendation.")
            return {
                "agent": "faq",
                "answer": "No reliable FAQ match found.",
                "confidence": confidence,
                "source": None,
                "matched_question": matched_question,
                "should_escalate": True,
            }


# Global Singleton FAQ Agent Instance
faq_agent = FAQAgent()


def process_faq_query(query: str, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Helper function interface for processing FAQ queries."""
    return faq_agent.process(query=query, threshold=threshold)
