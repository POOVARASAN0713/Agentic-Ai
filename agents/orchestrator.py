"""LangGraph orchestration for the multi-agent customer support workflow."""

from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.escalation_agent import process_escalation_query
from agents.faq_agent import detect_query_language, process_faq_query
from agents.guardrails import validate_response
from agents.rag_agent import process_rag_query
from utils.document_loader import chunk_documents, load_faqs, load_policy_documents
from utils.logging_utils import get_logger
from utils.vectorstore import vector_store

logger = get_logger("SupportOrchestrator")

GREETING_RESPONSES = {
    "hi": "Hi! How can I help you with your online order, delivery, return, refund, or account?",
    "hello": "Hello! How can I help you with your online order, delivery, return, refund, or account?",
    "hey": "Hey! How can I help you today?",
}

LOCALIZED_GREETING_RESPONSES = {
    "ta": "வணக்கம்! உங்கள் ஆர்டர், டெலிவரி, ரிட்டர்ன், ரீஃபண்ட் அல்லது கணக்கு தொடர்பாக நான் எவ்வாறு உதவலாம்?",
    "hi": "नमस्ते! मैं आपके ऑर्डर, डिलीवरी, रिटर्न, रिफंड या अकाउंट में कैसे मदद कर सकता हूँ?",
}

LOCALIZED_GREETING_KEYS = {
    "வணக்கம்": "ta",
    "नमस्ते": "hi",
}


class SupportState(TypedDict, total=False):
    query: str
    route: str
    response: Dict[str, Any]
    trace: List[str]
    escalation_reason: Optional[str]
    confidence: float


class SupportOrchestrator:
    """Routes a support query through specialist agents and final guardrails."""

    def __init__(
        self,
        faq_processor: Callable[..., Dict[str, Any]] = process_faq_query,
        rag_processor: Callable[..., Dict[str, Any]] = process_rag_query,
        escalation_processor: Callable[..., Dict[str, Any]] = process_escalation_query,
    ):
        self.faq_processor = faq_processor
        self.rag_processor = rag_processor
        self.escalation_processor = escalation_processor
        self._knowledge_base_ready = False
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(SupportState)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("faq", self._faq_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("escalation", self._escalation_node)
        workflow.add_node("guardrails", self._guardrails_node)
        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {"faq": "faq", "rag": "rag", "escalation": "escalation"},
        )
        workflow.add_conditional_edges(
            "faq",
            self._route_after_faq,
            {"rag": "rag", "guardrails": "guardrails"},
        )
        workflow.add_conditional_edges(
            "rag",
            self._route_after_rag,
            {"escalation": "escalation", "guardrails": "guardrails"},
        )
        workflow.add_edge("escalation", "guardrails")
        workflow.add_edge("guardrails", END)
        return workflow.compile()

    def _classify_node(self, state: SupportState) -> Dict[str, Any]:
        query = state["query"]
        from agents.escalation_agent import escalation_agent

        self._ensure_knowledge_base()
        if vector_store.faq_collection.count() == 0:
            return {
                "route": "rag",
                "trace": ["classify:faq_unavailable", "classify:policy_candidate"],
            }

        query_lower = query.lower()
        policy_markers = (
            "policy",
            "warranty",
            "coverage",
            "eligibility",
            "rules",
            "terms",
            "time limit",
            "timeframe",
            "required documents",
            "documents are required",
            "requirements",
            "exclusions",
            "working hours",
            "support hours",
        )
        is_policy_question = any(marker in query_lower for marker in policy_markers)
        is_support_availability_question = (
            "support" in query_lower
            and any(term in query_lower for term in ("ai", "human", "duration", "available", "hours", "system"))
        )
        if is_policy_question:
            return {"route": "rag", "trace": ["classify:policy_candidate"]}
        if is_support_availability_question and "policy" in query_lower:
            return {"route": "rag", "trace": ["classify:policy_candidate"]}
        if is_support_availability_question:
            return {"route": "faq", "trace": ["classify:support_availability"]}
        if escalation_agent.check_explicit_human_request(query):
            return {
                "route": "escalation",
                "escalation_reason": "Explicit human support representative requested",
                "trace": ["classify:explicit_human_request"],
            }
        if escalation_agent.check_sensitive_topic(query):
            return {
                "route": "escalation",
                "escalation_reason": "Sensitive / high-priority issue detected",
                "trace": ["classify:sensitive_topic"],
            }
        return {"route": "faq", "trace": ["classify:faq_candidate"]}

    def _ensure_knowledge_base(self) -> None:
        if self._knowledge_base_ready:
            return
        faqs = load_faqs()
        documents = load_policy_documents()
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        vector_store.index_faqs(faqs, force_reindex=False)
        vector_store.index_documents(chunks, force_reindex=False)
        self._knowledge_base_ready = True

    def _faq_node(self, state: SupportState) -> Dict[str, Any]:
        self._ensure_knowledge_base()
        result = self.faq_processor(state["query"])
        return {
            "route": "faq",
            "response": result,
            "confidence": result.get("confidence", 0.0),
            "trace": state.get("trace", []) + ["faq"],
        }

    def _rag_node(self, state: SupportState) -> Dict[str, Any]:
        result = self.rag_processor(state["query"])
        return {
            "route": "rag",
            "response": result,
            "confidence": result.get("confidence", 0.0),
            "trace": state.get("trace", []) + ["rag"],
        }

    def _escalation_node(self, state: SupportState) -> Dict[str, Any]:
        result = self.escalation_processor(
            state["query"],
            reason=state.get("escalation_reason"),
            confidence=state.get("confidence", 0.0),
        )
        return {
            "route": "escalation",
            "response": result,
            "trace": state.get("trace", []) + ["escalation"],
        }

    @staticmethod
    def _route_after_classification(state: SupportState) -> str:
        return state["route"]

    @staticmethod
    def _route_after_faq(state: SupportState) -> str:
        return "rag" if state["response"].get("should_escalate") else "guardrails"

    @staticmethod
    def _route_after_rag(state: SupportState) -> str:
        return "escalation" if state["response"].get("should_escalate") else "guardrails"

    def _guardrails_node(self, state: SupportState) -> Dict[str, Any]:
        response = validate_response(state["query"], state["response"])
        response["route"] = state.get("route", response.get("agent", "unknown"))
        response["trace"] = state.get("trace", []) + ["guardrails"]
        return {"response": response, "trace": response["trace"]}

    def process(self, query: str) -> Dict[str, Any]:
        """Process one query and return the final auditable response payload."""
        clean_query = (query or "").strip()
        if not clean_query:
            return self._guardrails_node(
                {
                    "query": "",
                    "route": "escalation",
                    "trace": ["classify:empty_query"],
                    "response": self.escalation_processor(
                        "", reason="Empty input", confidence=0.0
                    ),
                }
            )["response"]
        language = detect_query_language(clean_query)
        normalized_query = clean_query.lower().rstrip("!?.").strip()
        greeting = GREETING_RESPONSES.get(normalized_query, "")
        greeting_language = LOCALIZED_GREETING_KEYS.get(normalized_query)
        if greeting_language:
            greeting = LOCALIZED_GREETING_RESPONSES[greeting_language]
        if greeting and language in LOCALIZED_GREETING_RESPONSES:
            greeting = LOCALIZED_GREETING_RESPONSES[language]
        if greeting:
            return self._guardrails_node(
                {
                    "query": clean_query,
                    "route": "general",
                    "trace": ["classify:greeting", "general"],
                    "response": {
                        "agent": "general",
                        "answer": greeting,
                        "confidence": 1.0,
                        "should_escalate": False,
                    },
                }
            )["response"]
        state = self.graph.invoke({"query": clean_query})
        return state["response"]


support_orchestrator = SupportOrchestrator()


def process_support_query(query: str) -> Dict[str, Any]:
    """Convenience interface for the application and frontend."""
    return support_orchestrator.process(query)
