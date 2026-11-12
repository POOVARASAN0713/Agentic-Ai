"""
Unit Tests for RAG Specialist Agent.
Tests document-grounded policy retrieval, source tracking, threshold filtering, and out-of-policy queries.
"""

import pytest
from utils.document_loader import load_policy_documents, chunk_documents
from utils.vectorstore import vector_store
from agents.rag_agent import _is_policy_query, rag_agent, process_rag_query


@pytest.fixture(scope="module", autouse=True)
def setup_rag_vectorstore():
    """Ensure vectorstore document collection is indexed before running tests."""
    raw_docs = load_policy_documents()
    chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=50)
    vector_store.index_documents(chunks, force_reindex=False)


def test_hardware_warranty_rag_query():
    """Test RAG retrieval for hardware warranty terms."""
    query = "What is the warranty coverage period for hardware products?"
    response = process_rag_query(query)

    assert response["agent"] == "rag"
    assert response["confidence"] >= 0.65
    assert len(response["sources"]) > 0
    assert "online_marketplace_policy.txt" in response["sources"]


def test_refund_policy_rag_query():
    """Test RAG retrieval for refund window policy."""
    query = "How many calendar days do I have to request a refund?"
    response = process_rag_query(query)

    assert response["agent"] == "rag"
    assert response["confidence"] >= 0.65
    assert "online_marketplace_policy.txt" in response["sources"]


def test_out_of_scope_rag_query():
    """Test low confidence handling for unmentioned out-of-scope query."""
    query = "How do I cultivate organic tomatoes in a greenhouse?"
    response = process_rag_query(query, threshold=0.65)

    assert response["agent"] == "rag"
    assert response["confidence"] < 0.65
    assert response["should_escalate"] is True
    assert "Sufficient information was not found" in response["answer"]


def test_empty_rag_query():
    """Test handling of empty query string."""
    response = process_rag_query("")

    assert response["agent"] == "rag"
    assert response["confidence"] == 0.0
    assert response["should_escalate"] is True
    assert response["answer"] == "Empty query provided."


def test_policy_query_detection():
    assert _is_policy_query("What are the international shipping rules?")
    assert _is_policy_query("What are the support working hours?")
    assert not _is_policy_query("Where is my order?")


def test_policy_fallback_answer_avoids_escalation_when_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "agents.rag_agent.llm_manager.invoke",
        lambda **_kwargs: {"status": "error", "content": ""},
    )
    response = process_rag_query("What are the international shipping rules?")

    assert response["should_escalate"] is False
    assert "7 to 14 business days" in response["answer"]
