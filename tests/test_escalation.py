"""
Unit Tests for Escalation Specialist Agent.
Tests explicit human request triggers, sensitive scenario escalation, low confidence fallback, and ticket creation.
"""

import pytest
from agents.escalation_agent import escalation_agent, process_escalation_query


def test_explicit_human_request():
    """Test escalation when user explicitly requests a human representative."""
    query = "Please connect me to a human representative right now."
    response = process_escalation_query(query)

    assert response["agent"] == "escalation"
    assert response["escalated"] is True
    assert "Explicit human support representative requested" in response["reason"]
    assert response["ticket_id"].startswith("TICKET-")
    assert "reference ticket is #" in response["answer"]


def test_sensitive_scenario_escalation():
    """Test escalation when sensitive security keywords are detected."""
    query = "My account was hacked and there is an unauthorized charge."
    response = process_escalation_query(query)

    assert response["agent"] == "escalation"
    assert response["escalated"] is True
    assert "Sensitive / high-priority issue detected" in response["reason"]
    assert response["ticket_id"] is not None


def test_low_confidence_fallback_escalation():
    """Test escalation when triggered by preceding agent low confidence."""
    query = "What is the policy for deep space exploration?"
    response = process_escalation_query(query, reason="Low retrieval confidence", confidence=0.25)

    assert response["agent"] == "escalation"
    assert response["escalated"] is True
    assert response["confidence"] == 0.25
    assert response["reason"] == "Low retrieval confidence"
