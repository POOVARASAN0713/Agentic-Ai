"""Benchmark multi-agent support query routing strategies against labelled support benchmark dataset."""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = Path(__file__).with_name("benchmark.json")

HUMAN_TERMS = ("human", "real person", "unauthorized", "hacked", "fraud", "scam")
POLICY_TERMS = ("refund", "warranty", "shipping", "delivery", "privacy", "policy")
OUT_OF_SCOPE_TERMS = ("deep space", "recipe", "chess")


def rule_based_baseline(query: str) -> str:
    """Classify common support intents without embeddings or API keys."""
    lowered = query.lower()
    if any(term in lowered for term in HUMAN_TERMS + OUT_OF_SCOPE_TERMS):
        return "escalation"
    if any(term in lowered for term in POLICY_TERMS):
        return "rag"
    return "faq"


def retrieval_similarity_route(query: str) -> str:
    """Route queries using pure vector similarity score thresholds (ChromaDB + SentenceTransformers)."""
    from utils.vectorstore import vector_store
    faq_matches = vector_store.search_faqs(query, top_k=1)
    doc_matches = vector_store.search_documents(query, top_k=1)

    faq_score = faq_matches[0]["similarity_score"] if faq_matches else 0.0
    doc_score = doc_matches[0]["similarity_score"] if doc_matches else 0.0

    if faq_score >= 0.75:
        return "faq"
    elif doc_score >= 0.40:
        return "rag"
    else:
        return "escalation"


def self_reported_llm_route(query: str) -> str:
    """Route queries using direct LLM prompt zero-shot intent classification."""
    from utils.llm import llm_manager
    prompt = (
        "You are an intent classifier for customer support. "
        "Classify the query into exactly one route: 'faq', 'rag', or 'escalation'.\n"
        "- 'faq': common user questions like password reset or account login\n"
        "- 'rag': detailed company policies like refunds, warranties, shipping\n"
        "- 'escalation': human agent requests, security/fraud issues, or out-of-scope/unusual topics\n\n"
        f"Query: {query}\n"
        "Return ONLY a JSON object: {\"route\": \"<category>\"}"
    )
    try:
        res = llm_manager.invoke(prompt)
        text = str(res.get("content", "")).lower()
        if "faq" in text:
            return "faq"
        elif "rag" in text:
            return "rag"
        elif "escalation" in text:
            return "escalation"
    except Exception:
        pass
    return rule_based_baseline(query)


def hybrid_rule_based_route(query: str) -> str:
    """Route queries using the full LangGraph Orchestrator (Rules + Vector DB + Fallbacks)."""
    from agents.orchestrator import process_support_query
    res = process_support_query(query)
    route = res.get("route", res.get("agent", "unknown"))
    return "faq" if route == "general" else route


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict[str, float]:
    """Calculate Accuracy, Macro Precision, and Macro Recall."""
    if labels is None:
        labels = ["faq", "rag", "escalation"]
    
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0

    precisions = []
    recalls = []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(macro_prec, 2),
        "recall": round(macro_rec, 2),
    }


def run_benchmark(all_strategies: bool = True) -> Dict[str, Any]:
    cases: List[Dict[str, str]] = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    y_true = [c["expected_route"] for c in cases]

    strategy_funcs = {
        "Self-reported LLM": self_reported_llm_route,
        "Retrieval similarity": retrieval_similarity_route,
        "Hybrid rule-based": hybrid_rule_based_route,
        "Rule-based baseline": rule_based_baseline,
    }

    report = {}
    for name, func in strategy_funcs.items():
        y_pred = [func(c["query"]) for c in cases]
        report[name] = compute_metrics(y_true, y_pred)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run support routing benchmark for all evaluation strategies.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    args = parser.parse_args()

    results = run_benchmark()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n| Strategy | Acc. | Prec. | Recall |")
        print("| --- | --- | --- | --- |")
        for strat, metrics in results.items():
            print(f"| {strat} | {metrics['accuracy']:.2f} | {metrics['precision']:.2f} | {metrics['recall']:.2f} |")
