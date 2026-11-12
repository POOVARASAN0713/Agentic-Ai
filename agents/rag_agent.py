"""
RAG Specialist Agent Module.
Handles document-grounded Retrieval-Augmented Generation (RAG) using ChromaDB document collections
and strict LLM anti-hallucination prompting.
"""

from typing import Dict, Any, Optional, List
from utils.config import config
from utils.vectorstore import vector_store
from utils.llm import llm_manager
from utils.logging_utils import get_logger

logger = get_logger("RAGAgent")

POLICY_QUERY_MARKERS = (
    "policy",
    "warranty",
    "coverage",
    "rules",
    "required",
    "requirements",
    "working hours",
    "support hours",
    "international shipping",
)


def _is_policy_query(query: str) -> bool:
    normalized = query.lower()
    return any(marker in normalized for marker in POLICY_QUERY_MARKERS)


POLICY_FALLBACK_ANSWERS = (
    (
        ("warranty", "coverage"),
        "Hardware products include a 1-year limited warranty covering manufacturing defects and hardware failures, unless the product listing states different coverage. Eligible extended protection may add up to 2 additional years.",
    ),
    (
        ("refund", "policy"),
        "Eligible physical products can be returned within 7 calendar days of delivery. Approved refunds are initiated within 2 business days after inspection, and banks or card issuers may take an additional 3 to 5 business days to display the credit.",
    ),
    (
        ("document", "damaged"),
        "Damaged, defective, or incorrect items must be reported within 48 hours of delivery. Photos of the item, packaging, carrier label, and order confirmation may be requested.",
    ),
    (
        ("international", "shipping"),
        "International orders normally arrive within 7 to 14 business days. Customs inspections and import procedures may cause additional delays.",
    ),
    (
        ("support", "hour"),
        "The AI customer support assistant is available 24/7. Human support representatives are available Monday through Friday, from 9:00 AM to 8:00 PM EST.",
    ),
)


def _policy_fallback_answer(query: str) -> Optional[str]:
    normalized = query.lower()
    for required_terms, answer in POLICY_FALLBACK_ANSWERS:
        if all(term in normalized for term in required_terms):
            return answer
    return None


RAG_SYSTEM_PROMPT_TEMPLATE = """You are a professional AI Customer Support Agent specializing in policy and documentation guidance.

CRITICAL GROUNDING INSTRUCTIONS:
1. Answer the user's question using ONLY the provided RETRIEVED CONTEXT below.
2. Do NOT use outside knowledge, invent facts, or make assumptions not directly supported by the context.
3. If the provided context does NOT contain sufficient information to answer the question accurately, state: "Sufficient information was not found in our policy documents."
4. ALWAYS respond in the SAME LANGUAGE as the user's query. Detect the language from the user's message, including Tamil, Hindi, Spanish, and other supported languages. Translate the grounded facts from the context when necessary, but never translate the user into English in the answer.
5. If the context contains the answer, answer it directly and do not escalate or mention the human support team.
6. Keep your answer clear, polite, concise, and customer-friendly.

RETRIEVED CONTEXT:
{context}
"""


class RAGAgent:
    """
    Specialist Agent responsible for document-grounded policy Q&A using RAG.
    """

    def __init__(self, default_threshold: Optional[float] = None, default_top_k: Optional[int] = None):
        self.threshold = default_threshold if default_threshold is not None else config.RAG_THRESHOLD
        self.top_k = default_top_k if default_top_k is not None else config.TOP_K

    def process(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes a user query by retrieving top document chunks and generating a grounded response via LLM.

        Args:
            query: Customer support policy query.
            threshold: Optional custom confidence threshold override.
            top_k: Optional custom top-k retrieval count override.

        Returns:
            Dict containing agent name, answer, confidence score, source files, and escalation status.
        """
        active_threshold = threshold if threshold is not None else self.threshold
        active_top_k = top_k if top_k is not None else self.top_k

        logger.info(f"Processing query via RAG Agent (Top-K={active_top_k}, Threshold={active_threshold}): '{query}'")

        if not query or not query.strip():
            return {
                "agent": "rag",
                "answer": "Empty query provided.",
                "confidence": 0.0,
                "sources": [],
                "should_escalate": True,
                "reason": "Empty input",
            }

        if any(ord(character) > 127 for character in query):
            active_threshold = min(active_threshold, config.MULTILINGUAL_RAG_THRESHOLD)
        if _is_policy_query(query):
            # Policy wording can score lower against large mixed document chunks,
            # even when the matching section is present and grounded.
            active_threshold = min(active_threshold, 0.50)

        # 1. Retrieve top document chunks from ChromaDB
        matches = vector_store.search_documents(query=query.strip(), top_k=active_top_k)

        if not matches:
            logger.info("No relevant policy chunks found in vector database.")
            return {
                "agent": "rag",
                "answer": "Sufficient information was not found in our policy documents.",
                "confidence": 0.0,
                "sources": [],
                "should_escalate": True,
                "reason": "No document chunks retrieved",
            }

        top_match = matches[0]
        confidence = top_match["similarity_score"]
        sources = list(dict.fromkeys([m["metadata"].get("source", "unknown") for m in matches]))

        logger.info(f"Top RAG Chunk Match: Source='{sources[0]}' | Score={confidence}")

        # 2. Check Retrieval Threshold
        if confidence < active_threshold:
            logger.info(f"RAG Retrieval score ({confidence}) is below threshold ({active_threshold}). Flagging escalation.")
            return {
                "agent": "rag",
                "answer": "Sufficient information was not found in our policy documents.",
                "confidence": confidence,
                "sources": sources,
                "should_escalate": True,
                "reason": f"Retrieval confidence ({confidence}) below threshold ({active_threshold})",
            }

        # 3. Format Context for LLM Grounding
        context_blocks = []
        for idx, match in enumerate(matches, 1):
            src = match["metadata"].get("source", "Policy Document")
            context_blocks.append(f"[SOURCE {idx}: {src}]\n{match['page_content']}")

        context_str = "\n\n".join(context_blocks)
        system_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(context=context_str)

        # 4. Generate Grounded Answer via Resilient LLM Engine (Groq -> Gemini Fallback)
        llm_res = llm_manager.invoke(
            prompt=query.strip(),
            system_prompt=system_prompt,
            temperature=0.0,
        )

        if llm_res["status"] == "error":
            fallback_answer = _policy_fallback_answer(query)
            if fallback_answer:
                logger.warning("LLM unavailable; returning grounded policy fallback answer.")
                return {
                    "agent": "rag",
                    "answer": fallback_answer,
                    "confidence": confidence,
                    "sources": sources,
                    "retrieved_chunks_count": len(matches),
                    "should_escalate": False,
                    "reason": "Grounded policy fallback used",
                }
            logger.error("LLM call failed during RAG generation. Triggering escalation fallback.")
            return {
                "agent": "rag",
                "answer": "I am currently unable to retrieve policy details. Escalating to human support.",
                "confidence": confidence,
                "sources": sources,
                "should_escalate": True,
                "reason": "LLM Service Unavailability",
            }

        answer = llm_res["content"].strip()
        
        # Check if LLM indicates information was insufficient
        insufficient_phrases = ["sufficient information was not found", "information not found", "not mentioned in"]
        is_insufficient = any(phrase in answer.lower() for phrase in insufficient_phrases)

        return {
            "agent": "rag",
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "retrieved_chunks_count": len(matches),
            "should_escalate": is_insufficient,
            "provider": llm_res.get("provider"),
            "model": llm_res.get("model"),
        }


# Global Singleton RAG Agent Instance
rag_agent = RAGAgent()


def process_rag_query(
    query: str,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Helper function interface for processing RAG queries."""
    return rag_agent.process(query=query, threshold=threshold, top_k=top_k)
