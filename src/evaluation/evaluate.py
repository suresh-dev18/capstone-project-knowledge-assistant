"""
Stage 6: Evaluation
----------------------
Runs a fixed set of test questions through the RAG chain and scores:
  1. Retrieval accuracy -- did the right source document get retrieved?
  2. Groundedness -- does the answer cite the retrieved chunks?
  3. Abstention correctness -- does the agent correctly refuse to answer
     questions with no support in the knowledge base?

Edit TEST_SET to match the documents in the target knowledge base.
"""
import os
import sys
import json
import re
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config
from src.rag.rag_chain import answer_question

TEST_SET = [
    {"question": "What is the company's remote work policy?",
     "expected_doc_id": "hr_policy_handbook", "should_have_answer": True},
    {"question": "How many vacation days do employees get per year?",
     "expected_doc_id": "hr_policy_handbook", "should_have_answer": True},
    {"question": "What is the CEO's personal cell phone number?",
     "expected_doc_id": None, "should_have_answer": False},
]

NO_ANSWER_PHRASE = "don't have enough information"


def score_retrieval(expected_doc_id, sources) -> bool:
    if expected_doc_id is None:
        return True
    return any(expected_doc_id in (s.get("doc_uri") or "") for s in sources)


def score_groundedness(answer: str) -> float:
    citation_pattern = re.findall(r"\[[\w\-]+_\d+\]", answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    return min(1.0, len(citation_pattern) / max(1, len(sentences)))


def score_abstention(should_have_answer: bool, answer: str) -> bool:
    abstained = NO_ANSWER_PHRASE in answer.lower()
    return (not abstained) if should_have_answer else abstained


def run():
    results = []
    for case in TEST_SET:
        response = answer_question(case["question"])
        answer = response["answer"]
        sources = response["sources"]

        results.append({
            "question": case["question"],
            "answer": answer,
            "retrieval_correct": score_retrieval(case["expected_doc_id"], sources),
            "groundedness_score": round(score_groundedness(answer), 2),
            "abstention_correct": score_abstention(case["should_have_answer"], answer),
        })

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(results),
        "retrieval_accuracy": round(sum(r["retrieval_correct"] for r in results) / len(results), 2),
        "avg_groundedness": round(sum(r["groundedness_score"] for r in results) / len(results), 2),
        "abstention_accuracy": round(sum(r["abstention_correct"] for r in results) / len(results), 2),
        "details": results,
    }

    print(json.dumps(summary, indent=2))

    os.makedirs("evaluation_reports", exist_ok=True)
    out_path = f"evaluation_reports/eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport written to {out_path}")
    return summary


if __name__ == "__main__":
    run()
