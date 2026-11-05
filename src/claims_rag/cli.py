"""Command line: index | retrieve | decide.

    python -m claims_rag.cli index
    python -m claims_rag.cli retrieve --claim C004
    python -m claims_rag.cli decide --claim C004
"""
import argparse

from claims_rag.decision.decide import decide
from claims_rag.decision.facts import compute_facts
from claims_rag.ingestion.index import build_index
from claims_rag.loader import find_claim
from claims_rag.retrieval.metrics import score_retrieval
from claims_rag.retrieval.retrieve import RetrievalResult, retrieve

DIVIDER = "=" * 70


def print_heading(title: str) -> None:
    print("\n" + DIVIDER)
    print(title)
    print(DIVIDER)


def print_retrieval_result(retrieval: RetrievalResult) -> None:
    """Show the search text and the retrieved chunks, best match first."""
    print(f"QUERY:\n{retrieval.query}\n")
    print("RETRIEVED (best match first; 16.1 is always included):")
    for position, chunk in enumerate(retrieval.chunks, start=1):
        # The first line of a stored chunk is its section title, so the second line is the clause text.
        clause_text_line = chunk.page_content.split("\n")[1]
        print(f"  {position:>2}. chunk={chunk.metadata['chunk_id']:<7} clauses={chunk.metadata['clause_ids']}")
        print(f"      {clause_text_line[:90]}")
    print(f"\nCONTEXT CLAUSE IDS: {sorted(retrieval.context_clause_ids)}")


def print_facts(fact_statements: list[str]) -> None:
    print("\nCOMPUTED FACTS:")
    for statement in fact_statements:
        print(f"  - {statement}")


def run_retrieve_command(claim_id: str) -> None:
    """Retrieve clauses for one claim and show how well the retrieval matched the labelled clauses."""
    claim, test_claim = find_claim(claim_id)
    retrieval = retrieve(claim)  # only the claim without answers is used for retrieval
    print_retrieval_result(retrieval)
    print_facts(compute_facts(claim).as_statements())

    # The answer key is used below only to score the retrieval, for inspection.
    score = score_retrieval(retrieval.context_clause_ids, test_claim.expected_clauses)
    print_heading("RETRIEVAL METRICS (vs labelled expected_clauses, for inspection only)")
    print(f"expected: {score.expected}")
    print(f"found:    {score.found}")
    print(f"missing:  {score.missing}")
    if score.recall is None:
        recall_text = "n/a"
    else:
        recall_text = f"{score.recall:.2f} ({len(score.found)}/{len(score.expected)})"
    print(f"recall:    {recall_text}")
    print(f"precision: {score.precision:.2f} ({len(score.found)}/{score.retrieved_count} retrieved clauses)")


def run_decide_command(claim_id: str) -> None:
    """Run the full pipeline on one claim, then show the expected answer for comparison."""
    claim, test_claim = find_claim(claim_id)
    result = decide(claim)  # only the claim without answers goes in
    print_retrieval_result(result.retrieval)
    print_facts(result.facts.as_statements())

    decision = result.decision
    print_heading("DECISION")
    print(f"decision:   {decision.decision}")
    print(f"payable:    {decision.payable_amount}")
    print(f"confidence: {decision.confidence}")
    print(f"citations:  {decision.citations}")
    if result.llm_decision is not None and result.routed_by_confidence:
        print(f"routed_by_confidence: LLM said '{result.llm_decision.decision}', changed to needs_review")
    print(f"\nreasoning:\n{decision.reasoning}")

    # If any attempt was rejected by validation, say so (the retry may still have succeeded).
    rejected_attempts = []
    for attempt in result.attempts:
        if attempt["validation_errors"]:
            rejected_attempts.append(attempt["validation_errors"])
    if rejected_attempts:
        print(f"\nvalidation problems seen: {rejected_attempts}")

    # The answer key comes last and is clearly separated, so it cannot be confused with the result.
    print_heading("EXPECTED (answer key, shown after the call, for comparison only)")
    print(f"decision: {test_claim.expected_decision}")
    print(f"payable:  {test_claim.expected_payable}")
    print(f"clauses:  {test_claim.expected_clauses}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="claims_rag")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("index", help="chunk the policy and build the Chroma index")

    retrieve_parser = subcommands.add_parser("retrieve", help="show retrieved chunks and computed facts")
    retrieve_parser.add_argument("--claim", required=True)

    decide_parser = subcommands.add_parser("decide", help="run the full pipeline")
    decide_parser.add_argument("--claim", required=True)

    args = parser.parse_args()
    if args.command == "index":
        print(f"Indexed {build_index()} chunks.")
    elif args.command == "retrieve":
        run_retrieve_command(args.claim)
    elif args.command == "decide":
        run_decide_command(args.claim)


if __name__ == "__main__":
    main()
