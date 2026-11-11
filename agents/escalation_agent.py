"""
Escalation Specialist Agent Module.
Handles query transfers to human support representatives for low-confidence queries,
explicit human handoff requests, or sensitive customer scenarios.
"""

import random
import time
from typing import Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger("EscalationAgent")

HUMAN_REQUEST_KEYWORDS = [
    "human", "agent", "representative", "real person", "person",
    "customer support executive", "customer service officer",
    "talk to someone", "speak to someone", "connect me", "escalate",
]

SENSITIVE_KEYWORDS = [
    "hacked", "stolen", "unauthorized", "fraud", "scam",
    "double charged", "legal action", "lawyer", "police", "court",
]


class EscalationAgent:
    """
    Specialist Agent responsible for managing human support handoffs and risk escalation.
    """

    def generate_ticket_id(self) -> str:
        """Generates a tracking ticket ID for human support queues."""
        random_num = random.randint(1000, 9999)
        return f"TICKET-{random_num}"

    def check_explicit_human_request(self, query: str) -> bool:
        """Checks if the query contains explicit requests for a human agent."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in HUMAN_REQUEST_KEYWORDS)

    def check_sensitive_topic(self, query: str) -> bool:
        """Checks if the query involves sensitive security or legal topics."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in SENSITIVE_KEYWORDS)

    def process(
        self,
        query: str,
        reason: Optional[str] = None,
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Processes an escalation trigger and constructs a human transfer response payload.

        Args:
            query: User customer support query.
            reason: Optional explicit reason for escalation.
            confidence: Confidence score of preceding agent attempt.

        Returns:
            Dict containing agent name, answer message, confidence, escalation status, and ticket ID.
        """
        logger.info(f"Processing Escalation Agent request for query: '{query}'")
        ticket_id = self.generate_ticket_id()

        # Determine escalation reason
        if self.check_explicit_human_request(query):
            determined_reason = "Explicit human support representative requested"
        elif self.check_sensitive_topic(query):
            determined_reason = "Sensitive / high-priority issue detected"
        elif reason:
            determined_reason = reason
        else:
            determined_reason = "Low retrieval confidence / Outside knowledge base"

        logger.info(f"Escalation Triggered: Reason='{determined_reason}' | Ticket={ticket_id}")

        answer_message = (
            f"I'm not confident that I can provide a completely accurate answer to your request. "
            f"I have escalated your query to a human support representative. "
            f"Your reference ticket is #{ticket_id}. Our team will assist you shortly."
        )

        return {
            "agent": "escalation",
            "answer": answer_message,
            "confidence": round(float(confidence), 4),
            "escalated": True,
            "reason": determined_reason,
            "ticket_id": ticket_id,
        }


# Global Singleton Escalation Agent Instance
escalation_agent = EscalationAgent()


def process_escalation_query(
    query: str,
    reason: Optional[str] = None,
    confidence: float = 0.0,
) -> Dict[str, Any]:
    """Helper function interface for processing escalation requests."""
    return escalation_agent.process(query=query, reason=reason, confidence=confidence)


def process_escalation(
    query: str,
    reason: Optional[str] = None,
    confidence: float = 0.0,
) -> Dict[str, Any]:
    """Public helper matching the Phase 6 escalation agent interface."""
    return escalation_agent.process(query=query, reason=reason, confidence=confidence)
