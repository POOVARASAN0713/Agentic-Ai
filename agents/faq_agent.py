"""
FAQ Specialist Agent Module.
Handles fast semantic similarity search against the ChromaDB FAQ collection,
applying strict confidence thresholding to return accurate answers or trigger escalation.
"""

from functools import lru_cache
from typing import Dict, Any, Optional
from utils.config import config
from utils.llm import llm_manager
from utils.vectorstore import vector_store
from utils.logging_utils import get_logger

logger = get_logger("FAQAgent")

FAQ_TRANSLATION_PROMPT = """You are a multilingual customer support translator.
Translate the verified FAQ answer into the exact language and writing system used by the user.
The user language is {language}. Do not answer in English unless the user language is English.
Preserve every product name, menu path, number, date, and quoted UI label accurately.
Do not add facts, explanations, or English text.
Return only the final customer-facing answer.

User question:
{query}

Verified FAQ answer:
{answer}
"""


def detect_query_language(query: str) -> str:
    """Return a lightweight language code without an extra model call."""
    if any("\u0900" <= character <= "\u097f" for character in query):
        return "hi"
    if any("\u0b80" <= character <= "\u0bff" for character in query):
        return "ta"
    normalized = query.lower()
    language_markers = {
        "es": ("qué", "cuál", "cómo", "dónde", "reembolso", "por favor"),
        "fr": ("bonjour", "merci", "comment", "remboursement"),
        "de": ("warum", "wie", "rückerstattung", "bitte"),
    }
    for language, markers in language_markers.items():
        if any(marker in normalized for marker in markers):
            return language
    return "en"


def _needs_translation(query: str) -> bool:
    """Detect whether the answer needs to be localized."""
    return detect_query_language(query) != "en"


LOCALIZED_FALLBACKS = {
    "ta": {
        "cancel": "எனது ஆர்டர்களுக்குச் சென்று செயலில் உள்ள ஆர்டரைத் தேர்ந்தெடுத்து, அது ஷிப்பிங் நிலைக்கு செல்லும் முன் ‘Cancel Order’ என்பதைக் கிளிக் செய்யவும்.",
        "password": "உள்நுழைவு திரையில் ‘Forgot Password’ என்பதைக் கிளிக் செய்து, உங்கள் பதிவு செய்யப்பட்ட மின்னஞ்சலுக்கு அனுப்பப்படும் வழிமுறைகளைப் பின்பற்றவும்.",
        "refund": "அங்கீகரிக்கப்பட்ட பணத்தைத் திரும்பப்பெறுதல் 3 முதல் 5 வேலை நாட்களுக்குள் உங்கள் அசல் பணம் செலுத்திய முறைக்குத் திருப்பி அனுப்பப்படும்.",
        "track": "உங்கள் ஆர்டர் கிடங்கிலிருந்து புறப்பட்டதும், நேரடி கேரியர் கண்காணிப்பு இணைப்புடன் மின்னஞ்சல் அனுப்பப்படும்.",
    },
    "hi": {
        "cancel": "My Orders में जाएं, सक्रिय ऑर्डर चुनें और शिपिंग चरण में जाने से पहले 'Cancel Order' पर क्लिक करें।",
        "password": "लॉगिन स्क्रीन पर 'Forgot Password' पर क्लिक करें और अपने पंजीकृत ईमेल पर भेजे गए निर्देशों का पालन करें।",
        "refund": "स्वीकृत रिफंड 3 से 5 कार्यदिवसों में आपकी मूल भुगतान विधि में भेज दिया जाएगा।",
        "track": "आपका ऑर्डर वेयरहाउस से भेजे जाने के बाद सीधे कैरियर ट्रैकिंग लिंक वाला ईमेल भेजा जाएगा।",
    },
}


def _localized_fallback(language: str, answer: str) -> str:
    """Keep common FAQ responses useful even when translation providers are unavailable."""
    answer_lower = answer.lower()
    for keyword, localized in LOCALIZED_FALLBACKS.get(language, {}).items():
        if keyword == "cancel" and ("cancel" in answer_lower or "my orders" in answer_lower):
            return localized
        if keyword == "password" and "password" in answer_lower:
            return localized
        if keyword == "refund" and "refund" in answer_lower:
            return localized
        if keyword == "track" and "tracking" in answer_lower:
            return localized
    return {
        "ta": "இந்த கேள்விக்கான சரிபார்க்கப்பட்ட தகவல் தற்போது கிடைக்கவில்லை. தயவுசெய்து ஆதரவு குழுவைத் தொடர்புகொள்ளவும்.",
        "hi": "इस प्रश्न के लिए सत्यापित जानकारी अभी उपलब्ध नहीं है। कृपया सहायता टीम से संपर्क करें।",
    }.get(language, answer)


def _uses_expected_script(text: str, language: str) -> bool:
    if language == "ta":
        return any("\u0b80" <= character <= "\u0bff" for character in text)
    if language == "hi":
        return any("\u0900" <= character <= "\u097f" for character in text)
    return True


@lru_cache(maxsize=256)
def _translate_cached(query: str, answer: str, language: str) -> str:
    result = llm_manager.invoke(
        prompt=(
            f"User language: {language}\n"
            f"User question: {query}\n\n"
            f"Verified answer to translate:\n{answer}"
        ),
        system_prompt=FAQ_TRANSLATION_PROMPT.format(
            query=query, answer=answer, language=language
        ),
        temperature=0.0,
    )
    translated = result.get("content", "").strip()
    if (
        result["status"] == "error"
        or not translated
        or not _uses_expected_script(translated, language)
    ):
        logger.warning("Translation did not preserve the requested language; using local FAQ fallback.")
        return _localized_fallback(language, answer)
    return translated


def _answer_in_query_language(query: str, answer: str) -> str:
    """Translate a verified FAQ answer while preserving the user's language."""
    language = detect_query_language(query)
    if language == "en":
        return answer
    return _translate_cached(query.strip(), answer, language)


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
        matches = vector_store.search_faqs(query=query.strip(), top_k=3)

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
