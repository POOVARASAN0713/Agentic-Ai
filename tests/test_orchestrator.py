from agents.orchestrator import SupportOrchestrator


def test_explicit_human_request_routes_directly_to_escalation():
    calls = []

    def faq(_query):
        calls.append("faq")
        return {"agent": "faq", "answer": "unused", "confidence": 1.0, "should_escalate": False}

    def rag(_query):
        calls.append("rag")
        return {"agent": "rag", "answer": "unused", "confidence": 1.0, "should_escalate": False}

    def escalation(query, reason=None, confidence=0.0):
        calls.append("escalation")
        return {
            "agent": "escalation",
            "answer": "Transferred to support.",
            "confidence": confidence,
            "escalated": True,
            "reason": reason,
            "ticket_id": "TICKET-1000",
        }

    response = SupportOrchestrator(faq, rag, escalation).process("I want a real person")

    assert response["route"] == "escalation"
    assert response["escalated"] is True
    assert calls == ["escalation"]
    assert response["guardrail_passed"] is True


def test_greeting_gets_general_response_without_escalation():
    response = SupportOrchestrator().process("Hello!")

    assert response["agent"] == "general"
    assert response["route"] == "general"
    assert response["should_escalate"] is False
    assert "How can I help" in response["answer"]


def test_support_system_question_is_not_treated_as_human_handoff():
    response = SupportOrchestrator().process("Is the support system AI or human?")

    assert response["agent"] == "faq"
    assert response["route"] == "faq"
    assert response["should_escalate"] is False
    assert "AI" in response["answer"]


def test_ai_support_policy_question_uses_rag():
    response = SupportOrchestrator().process("What is the policy for AI support availability?")

    assert response["agent"] == "rag"
    assert response["route"] == "rag"
    assert response["should_escalate"] is False
    assert "24 hours" in response["answer"]


def test_faq_success_stops_before_rag():
    calls = []

    def faq(_query):
        calls.append("faq")
        return {"agent": "faq", "answer": "Password reset steps", "confidence": 0.95, "should_escalate": False}

    def rag(_query):
        calls.append("rag")
        raise AssertionError("RAG should not run after a confident FAQ answer")

    def escalation(*_args, **_kwargs):
        raise AssertionError("Escalation should not run after a confident FAQ answer")

    response = SupportOrchestrator(faq, rag, escalation).process("How do I reset my password?")

    assert response["agent"] == "faq"
    assert response["route"] == "faq"
    assert calls == ["faq"]
    assert response["guardrail_passed"] is True


def test_low_confidence_path_uses_rag_then_escalates():
    calls = []

    def faq(_query):
        calls.append("faq")
        return {"agent": "faq", "answer": "No match", "confidence": 0.2, "should_escalate": True}

    def rag(_query):
        calls.append("rag")
        return {
            "agent": "rag",
            "answer": "No policy match",
            "confidence": 0.1,
            "should_escalate": True,
            "reason": "Low retrieval confidence",
        }

    def escalation(query, reason=None, confidence=0.0):
        calls.append("escalation")
        return {
            "agent": "escalation",
            "answer": "Transferred to support.",
            "confidence": confidence,
            "escalated": True,
            "reason": reason,
            "ticket_id": "TICKET-1001",
        }

    response = SupportOrchestrator(faq, rag, escalation).process("Unknown policy question")

    assert calls == ["faq", "rag", "escalation"]
    assert response["agent"] == "escalation"
    assert response["route"] == "escalation"
    assert response["guardrail_passed"] is True
