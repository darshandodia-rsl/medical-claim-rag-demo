"""Loading the test claims.

This file is the only place where the answer key is separated from the claim inputs.
"""
import json
from pathlib import Path

from claims_rag.config import TEST_CLAIMS_PATH
from claims_rag.schemas import Claim, EvalClaim


def load_test_claims(path: Path = TEST_CLAIMS_PATH) -> list[EvalClaim]:
    """Read every test claim from the JSON file, answer key included (for comparing results by eye)."""
    raw_claims = json.loads(path.read_text())
    return [EvalClaim(**raw_claim) for raw_claim in raw_claims]


def strip_answers(test_claim: EvalClaim) -> Claim:
    """Copy only the 18 claim fields and leave the answer fields behind.

    The result is the only kind of object decide() accepts, so the answers can never reach the LLM.
    """
    claim_field_names = set(Claim.model_fields)
    claim_values = test_claim.model_dump(include=claim_field_names)
    return Claim(**claim_values)


def find_claim(claim_id: str, path: Path = TEST_CLAIMS_PATH) -> tuple[Claim, EvalClaim]:
    """Look up one test claim by ID, e.g. "C004".

    Returns two versions: the clean Claim to run through the pipeline, and the full EvalClaim
    (with answers) to compare the result against afterwards.
    """
    for test_claim in load_test_claims(path):
        if test_claim.claim_id == claim_id:
            return strip_answers(test_claim), test_claim
    raise KeyError(f"No test claim with ID {claim_id}")
