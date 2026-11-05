"""Measure retrieval on all the test claims and print the average precision and recall.

    python -m claims_rag.retrieval.eval_retrieval

It calls retrieve() exactly as the real pipeline does, so any change to retrieval shows up in these numbers.
No LLM is called, so it costs nothing. It reads the answer key (expected_clauses) only to score the results;
the decision code never imports this file.
"""
import json
from datetime import datetime, timezone

from claims_rag.config import EMBED_MODEL, RETRIEVAL_EVAL_LOG_PATH
from claims_rag.ingestion.index import open_vector_store
from claims_rag.loader import load_test_claims, strip_answers
from claims_rag.retrieval.metrics import RetrievalScore, ScoreSummary, score_retrieval, summarize_scores
from claims_rag.retrieval.retrieve import TOP_K, retrieve


def score_every_claim() -> tuple[list[RetrievalScore], list[dict]]:
    """Run retrieval for each test claim, print one row per claim, and return the scores and a row per claim."""
    vector_store = open_vector_store()  # opened once and reused; retrieval itself is local and free
    scores = []
    rows_for_log = []

    print(f"{'claim':<6} {'recall':>8} {'precision':>10} {'retrieved':>10}  missing")
    for test_claim in load_test_claims():
        claim = strip_answers(test_claim)  # retrieval only ever sees the claim without answers
        retrieval = retrieve(claim, vector_store)
        score = score_retrieval(retrieval.context_clause_ids, test_claim.expected_clauses)
        scores.append(score)

        recall_text = "n/a" if score.recall is None else f"{score.recall:.2f}"
        missing_text = score.missing if score.missing else ""
        print(f"{test_claim.claim_id:<6} {recall_text:>8} {score.precision:>10.2f} {score.retrieved_count:>10}  {missing_text}")
        rows_for_log.append(
            {"claim_id": test_claim.claim_id, "recall": score.recall, "precision": score.precision, "missing": score.missing}
        )

    return scores, rows_for_log


def print_summary(summary: ScoreSummary) -> None:
    print("\n" + "=" * 60)
    print(f"RETRIEVAL ({summary.claims} claims, top_k={TOP_K}, embed={EMBED_MODEL})")
    print("=" * 60)
    print(f"mean recall:     {summary.mean_recall:.3f}")
    print(f"mean precision:  {summary.mean_precision:.3f}")
    print(f"full recall:     {summary.full_recall}/{summary.claims} claims")
    print(f"zero recall:     {summary.zero_recall}/{summary.claims} claims")
    print(f"avg clauses sent to LLM: {summary.mean_retrieved:.1f}")


def save_run_to_history(summary: ScoreSummary, rows_for_log: list[dict]) -> None:
    """Append this run to the history file, one line per run, so runs before and after a change can be compared."""
    history_entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "top_k": TOP_K,
        "embed_model": EMBED_MODEL,
        "claims": summary.claims,
        "mean_recall": summary.mean_recall,
        "mean_precision": summary.mean_precision,
        "full_recall": summary.full_recall,
        "zero_recall": summary.zero_recall,
        "mean_retrieved": summary.mean_retrieved,
        "per_claim": rows_for_log,
    }
    RETRIEVAL_EVAL_LOG_PATH.parent.mkdir(exist_ok=True)
    with RETRIEVAL_EVAL_LOG_PATH.open("a") as history_file:
        history_file.write(json.dumps(history_entry) + "\n")
    print(f"\nappended to logs/{RETRIEVAL_EVAL_LOG_PATH.name}")


def main() -> None:
    scores, rows_for_log = score_every_claim()
    summary = summarize_scores(scores)
    print_summary(summary)
    save_run_to_history(summary, rows_for_log)


if __name__ == "__main__":
    main()
