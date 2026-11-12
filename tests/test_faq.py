"""
Unit Tests for FAQ Specialist Agent.
Tests exact query matching, paraphrased semantic search, low confidence rejection, and empty inputs.
"""

import pytest
from utils.document_loader import load_faqs
from utils.vectorstore import vector_store
from agents.faq_agent import faq_agent, process_faq_query


@pytest.fixture(scope="module", autouse=True)
def setup_faq_vectorstore():
    """Ensure vectorstore FAQ collection is indexed before running tests."""
    faqs = load_faqs()
    vector_store.index_faqs(faqs, force_reindex=False)


def test_exact_faq_match():
    """Test exact match for a known FAQ question."""
    query = "How do I reset my password?"
    response = process_faq_query(query)

    assert response["agent"] == "faq"
    assert response["confidence"] >= 0.75
    assert response["should_escalate"] is False
    assert "Forgot Password" in response["answer"]
    assert response["source"] is not None


def test_paraphrased_faq_match():
    """Test semantic matching for a paraphrased query."""
    query = "What is the process to change my account password?"
    response = process_faq_query(query)

    assert response["agent"] == "faq"
    assert response["confidence"] > 0.60
    assert response["matched_question"] == "How do I reset my password?"


def test_out_of_scope_query():
    """Test low confidence handling for an out-of-scope query."""
    query = "What is the speed of light in vacuum?"
    response = process_faq_query(query, threshold=0.75)

    assert response["agent"] == "faq"
    assert response["confidence"] < 0.75
    assert response["should_escalate"] is True
    assert response["answer"] == "No reliable FAQ match found."


def test_empty_query():
    """Test handling of empty string query."""
    response = process_faq_query("")

    assert response["agent"] == "faq"
    assert response["confidence"] == 0.0
    assert response["should_escalate"] is True
    assert response["answer"] == "Empty query provided."


def test_non_english_faq_answer_is_translated(monkeypatch):
    def translated_answer(**_kwargs):
        return {"status": "success", "content": "உங்கள் கடவுச்சொல்லை மீட்டமைக்கலாம்."}

    monkeypatch.setattr("agents.faq_agent.llm_manager.invoke", translated_answer)
    response = process_faq_query("கடவுச்சொல்லை எவ்வாறு மாற்றுவது?")

    assert response["should_escalate"] is False
    assert response["answer"] == "உங்கள் கடவுச்சொல்லை மீட்டமைக்கலாம்."


def test_non_english_faq_never_returns_english_translation(monkeypatch):
    def english_answer(**_kwargs):
        return {"status": "success", "content": "Go to My Orders and click Cancel Order."}

    monkeypatch.setattr("agents.faq_agent.llm_manager.invoke", english_answer)
    response = process_faq_query("எனது ஆர்டரை ரத்து செய்வது எப்படி?")

    assert response["should_escalate"] is False
    assert any("\u0b80" <= character <= "\u0bff" for character in response["answer"])
