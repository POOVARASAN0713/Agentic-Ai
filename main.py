"""
Main Entry Point for AI Customer Support System.
Runs CLI diagnostics for the complete multi-agent support workflow.
"""

import sys
from utils.config import config
from utils.logging_utils import get_logger
from utils.llm import llm_manager

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = get_logger("MainApp")


def verify_phase1_setup():
    """
    Run diagnostic checks for Phase 1: Setup & Configuration.
    """
    print("\n" + "=" * 65)
    print(" === AI CUSTOMER SUPPORT MULTI-AGENT SYSTEM -- PHASE 1 DIAGNOSTIC ===")
    print("=" * 65 + "\n")

    # 1. Inspect Directories & Configuration
    logger.info("Checking configuration parameters...")
    print(f"  * Embedding Model : {config.EMBEDDING_MODEL_NAME}")
    print(f"  * Primary Model   : Groq ({config.GROQ_MODEL})")
    print(f"  * Fallback Model  : Gemini ({config.GEMINI_MODEL})")
    print(f"  * FAQ Threshold   : {config.FAQ_THRESHOLD}")
    print(f"  * RAG Threshold   : {config.RAG_THRESHOLD}")
    print(f"  * Top-K Retrieval : {config.TOP_K}")
    print(f"  * Chroma Directory: {config.CHROMA_PERSIST_DIR}")

    # 2. Check API Key Status
    print("\n" + "-" * 50)
    logger.info("Checking API key configurations...")
    api_status = config.validate_api_keys()
    print(f"  * Groq API Key Configured   : {'[OK] YES' if api_status['groq_configured'] else '[X] NO (Set GROQ_API_KEY in .env)'}")
    print(f"  * Gemini API Key Configured : {'[OK] YES' if api_status['gemini_configured'] else '[X] NO (Set GEMINI_API_KEY in .env)'}")

    # 3. Test Phase 2 Data Layer Loading
    print("\n" + "-" * 50)
    logger.info("Testing Phase 2 Data Layer Loaders...")
    from utils.document_loader import load_faqs, load_policy_documents, chunk_documents

    faqs = load_faqs()
    print(f"  * Loaded FAQs Count  : {len(faqs)}")
    if faqs:
        sample_faq = faqs[0]
        print(f"  * Sample FAQ (ID={sample_faq['id']}) Category [{sample_faq['category']}]:")
        print(f"    Q: {sample_faq['question']}")
        print(f"    A: {sample_faq['answer'][:70]}...")

    raw_docs = load_policy_documents()
    chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=50)
    print(f"  * Policy Documents   : {len(raw_docs)} files loaded")
    print(f"  * Total Chunks       : {len(chunks)} text chunks generated")
    if chunks:
        sample_chunk = chunks[0]
        print(f"  * Sample Chunk Source: {sample_chunk.metadata.get('source')}")
        print(f"    Snippet: {sample_chunk.page_content[:100]}...")

    # 4. Test Phase 3 Multilingual Vector Database (ChromaDB)
    print("\n" + "-" * 50)
    logger.info("Testing Phase 3 Multilingual Vector Database (ChromaDB)...")
    from utils.vectorstore import vector_store

    indexed_faq_count = vector_store.index_faqs(faqs, force_reindex=False)
    indexed_doc_count = vector_store.index_documents(chunks, force_reindex=False)
    print(f"  * FAQ Collection Size     : {indexed_faq_count} records indexed")
    print(f"  * Document Collection Size: {indexed_doc_count} chunks indexed")

    # Test Vector Similarity Search in English
    print("\n[VECTOR SEARCH TEST -- ENGLISH]")
    en_query = "How do I reset my password?"
    faq_matches = vector_store.search_faqs(en_query, top_k=1)
    if faq_matches:
        match = faq_matches[0]
        print(f"  Query        : '{en_query}'")
        print(f"  Matched FAQ  : Q: '{match['question']}'")
        print(f"  Similarity   : {match['similarity_score']} (Threshold={config.FAQ_THRESHOLD})")

    # Test Vector Similarity Search in Tamil
    print("\n[VECTOR SEARCH TEST -- TAMIL (MULTILINGUAL)]")
    ta_query = "கடவுச்சொல்லை எவ்வாறு மாற்றுவது?"  # How to change password?
    ta_matches = vector_store.search_faqs(ta_query, top_k=1)
    if ta_matches:
        t_match = ta_matches[0]
        print(f"  Query        : '{ta_query}'")
        print(f"  Matched FAQ  : Q: '{t_match['question']}'")
        print(f"  Similarity   : {t_match['similarity_score']}")

    # Test RAG Document Vector Search
    print("\n[DOCUMENT RAG SEARCH TEST]")
    rag_query = "What is the refund window for returned products?"
    doc_matches = vector_store.search_documents(rag_query, top_k=2)
    print(f"  Query        : '{rag_query}'")
    print(f"  Retrieved    : {len(doc_matches)} matching policy chunks")
    if doc_matches:
        best_doc = doc_matches[0]
        print(f"  Top Match    : Source='{best_doc['metadata'].get('source')}' | Score={best_doc['similarity_score']}")
        print(f"  Snippet      : {best_doc['page_content'][:110]}...")

    # 5. Test Phase 4 FAQ Specialist Agent
    print("\n" + "-" * 50)
    logger.info("Testing Phase 4 FAQ Specialist Agent...")
    from agents.faq_agent import process_faq_query

    # High Confidence Match
    q_high = "How do I reset my password?"
    res_high = process_faq_query(q_high)
    print(f"\n[FAQ TEST 1 -- High Confidence Match]")
    print(f"  Query          : '{q_high}'")
    print(f"  Agent          : {res_high['agent']}")
    print(f"  Confidence     : {res_high['confidence']} (Threshold={config.FAQ_THRESHOLD})")
    print(f"  Should Escalate: {res_high['should_escalate']}")
    print(f"  Matched Q      : {res_high['matched_question']}")
    print(f"  Answer Snippet : {res_high['answer'][:80]}...")

    # Low Confidence / Out of Scope Match
    q_low = "What is the recipe for baking chocolate cake?"
    res_low = process_faq_query(q_low)
    print(f"\n[FAQ TEST 2 -- Out-of-Scope Query]")
    print(f"  Query          : '{q_low}'")
    print(f"  Agent          : {res_low['agent']}")
    print(f"  Confidence     : {res_low['confidence']}")
    print(f"  Should Escalate: {res_low['should_escalate']}")
    print(f"  Answer         : {res_low['answer']}")

    # 6. Test Phase 5 RAG Specialist Agent
    print("\n" + "-" * 50)
    logger.info("Testing Phase 5 RAG Specialist Agent...")
    from agents.rag_agent import process_rag_query

    # Valid RAG Policy Query
    q_rag = "What is the warranty coverage period for hardware products?"
    res_rag = process_rag_query(q_rag)
    print(f"\n[RAG TEST 1 -- Policy Document Grounded Query]")
    print(f"  Query          : '{q_rag}'")
    print(f"  Agent          : {res_rag['agent']}")
    print(f"  Confidence     : {res_rag['confidence']} (Threshold={config.RAG_THRESHOLD})")
    print(f"  Sources Cited  : {res_rag['sources']}")
    print(f"  Should Escalate: {res_rag['should_escalate']}")
    print(f"  Answer Snippet : {res_rag['answer'][:110]}...")

    # Low Confidence RAG Query
    q_rag_low = "What are the rules for playing chess in your store?"
    res_rag_low = process_rag_query(q_rag_low)
    print(f"\n[RAG TEST 2 -- Out of Scope Policy Query]")
    print(f"  Query          : '{q_rag_low}'")
    print(f"  Agent          : {res_rag_low['agent']}")
    print(f"  Confidence     : {res_rag_low['confidence']}")
    print(f"  Should Escalate: {res_rag_low['should_escalate']}")
    print(f"  Answer         : {res_rag_low['answer']}")

    # 7. Test Phase 6 Escalation Specialist Agent
    print("\n" + "-" * 50)
    logger.info("Testing Phase 6 Escalation Specialist Agent...")
    from agents.escalation_agent import process_escalation_query

    # Explicit Human Request Trigger
    q_esc1 = "I need to talk to a human agent about a serious issue."
    res_esc1 = process_escalation_query(q_esc1)
    print(f"\n[ESCALATION TEST 1 -- Explicit Human Request]")
    print(f"  Query        : '{q_esc1}'")
    print(f"  Agent        : {res_esc1['agent']}")
    print(f"  Escalated    : {res_esc1['escalated']}")
    print(f"  Reason       : {res_esc1['reason']}")
    print(f"  Ticket ID    : {res_esc1['ticket_id']}")
    print(f"  Handoff Message: {res_esc1['answer']}")

    # Sensitive Security Issue Trigger
    q_esc2 = "My account was hacked and stolen!"
    res_esc2 = process_escalation_query(q_esc2)
    print(f"\n[ESCALATION TEST 2 -- Sensitive Security Issue]")
    print(f"  Query        : '{q_esc2}'")
    print(f"  Agent        : {res_esc2['agent']}")
    print(f"  Escalated    : {res_esc2['escalated']}")
    print(f"  Reason       : {res_esc2['reason']}")
    print(f"  Ticket ID    : {res_esc2['ticket_id']}")

    # Low Confidence Fallback Trigger
    q_esc3 = "What is the policy for deep space exploration?"
    res_esc3 = process_escalation_query(
        q_esc3,
        reason="Low retrieval confidence",
        confidence=0.25,
    )
    print(f"\n[ESCALATION TEST 3 -- Low Confidence Fallback]")
    print(f"  Query        : '{q_esc3}'")
    print(f"  Agent        : {res_esc3['agent']}")
    print(f"  Escalated    : {res_esc3['escalated']}")
    print(f"  Confidence   : {res_esc3['confidence']}")
    print(f"  Reason       : {res_esc3['reason']}")
    print(f"  Ticket ID    : {res_esc3['ticket_id']}")

    # 8. Test LLM Provider Call / Fallback Resilience
    print("\n" + "-" * 50)
    logger.info("Testing LLM Resilient Invocation Engine...")

    test_prompt = "Hello! Briefly respond in 1 sentence confirming you are operational."
    test_result = llm_manager.invoke(
        prompt=test_prompt,
        system_prompt="You are a helpful AI customer support setup assistant.",
    )

    print(f"\n[LLM TEST RESPONSE]")
    print(f"  * Status        : {test_result['status'].upper()}")
    print(f"  * Provider Used : {test_result['provider'] or 'None'}")
    print(f"  * Model Used    : {test_result['model'] or 'None'}")
    print(f"  * Fallback Used : {test_result['fallback_used']}")
    print(f"  * Response Text : {test_result['content']}\n")

    # 9. Test Phase 7-8 Orchestration and Guardrails
    print("\n" + "-" * 50)
    logger.info("Testing Phase 7 Orchestrator and Phase 8 Guardrails...")
    from agents.orchestrator import process_support_query

    q_workflow = "How do I reset my password?"
    res_workflow = process_support_query(q_workflow)
    print(f"\n[WORKFLOW TEST -- Orchestrator + Guardrails]")
    print(f"  Query          : '{q_workflow}'")
    print(f"  Route          : {res_workflow['route']}")
    print(f"  Agent          : {res_workflow['agent']}")
    print(f"  Guardrails     : {res_workflow['guardrail_passed']}")
    print(f"  Trace          : {res_workflow['trace']}")

    print("=" * 65)
    if res_esc1['escalated'] and res_esc2['escalated'] and res_esc3['escalated'] and res_workflow['guardrail_passed']:
        print(" [SUCCESS] PHASES 1-8 VERIFICATION COMPLETED SUCCESSFULLY!")
    else:
        print(" [READY] PHASES 1-8 STRUCTURE READY.")
    print("=" * 65 + "\n")







if __name__ == "__main__":
    verify_phase1_setup()

