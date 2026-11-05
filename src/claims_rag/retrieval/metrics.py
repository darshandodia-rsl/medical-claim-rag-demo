"""Retrieval precision and recall, measured against the labelled expected clauses.

This reads the answer key, so it is only used for inspection (the `retrieve` command and eval_retrieval.py).
retrieve.py and decide.py never import it.
"""
from dataclasses import dataclass


@dataclass
class RetrievalScore:
    """How well retrieval did for ONE claim."""

    expected: list[str]  # the clauses the labeller says decide this claim
    found: list[str]  # expected clauses that were retrieved
    missing: list[str]  # expected clauses that were not retrieved
    retrieved_count: int  # how many clauses were retrieved in total
    precision: float  # found / retrieved: how much of what we retrieved was needed
    recall: float | None  # found / expected: how much of what was needed we retrieved (None if no labels)


def score_retrieval(retrieved_clause_ids: set[str], expected_clauses: list[str]) -> RetrievalScore:
    """Compare the clauses retrieved for a claim with the clauses the labeller expected.

    "Retrieved" means every clause ID inside the chunks sent to the LLM, including sub-clauses that were merged
    into a chunk and the always-added limits clause 16.1. Precision is low by design: about 13 clauses come
    back for 2 or 3 expected ones, so it is most useful for comparing two versions of retrieval.
    """
    # Remove duplicate labels but keep their order.
    expected = []
    for clause_id in expected_clauses:
        if clause_id not in expected:
            expected.append(clause_id)

    found = []
    missing = []
    for clause_id in expected:
        if clause_id in retrieved_clause_ids:
            found.append(clause_id)
        else:
            missing.append(clause_id)

    retrieved_count = len(retrieved_clause_ids)
    precision = len(found) / retrieved_count if retrieved_count > 0 else 0.0
    recall = len(found) / len(expected) if len(expected) > 0 else None

    return RetrievalScore(expected, found, missing, retrieved_count, precision, recall)


@dataclass
class ScoreSummary:
    """The scores averaged over many claims."""

    claims: int
    mean_recall: float  # averaged over the claims that have expected clauses
    mean_precision: float
    full_recall: int  # number of claims where every expected clause was retrieved
    zero_recall: int  # number of claims where none was
    mean_retrieved: float  # average number of clauses sent to the LLM (a rough measure of prompt size)


def summarize_scores(scores: list[RetrievalScore]) -> ScoreSummary:
    """Average the per-claim scores. Each claim counts equally, however many expected clauses it has."""
    # Claims without labels have no recall, so they are left out of the recall and precision averages.
    labelled_scores = [score for score in scores if score.recall is not None]

    full_recall = 0
    zero_recall = 0
    for score in labelled_scores:
        if score.recall == 1.0:
            full_recall += 1
        if score.recall == 0.0:
            zero_recall += 1

    total_recall = sum(score.recall for score in labelled_scores)
    total_precision = sum(score.precision for score in labelled_scores)
    total_retrieved = sum(score.retrieved_count for score in scores)

    return ScoreSummary(
        claims=len(scores),
        mean_recall=total_recall / len(labelled_scores),
        mean_precision=total_precision / len(labelled_scores),
        full_recall=full_recall,
        zero_recall=zero_recall,
        mean_retrieved=total_retrieved / len(scores),
    )
