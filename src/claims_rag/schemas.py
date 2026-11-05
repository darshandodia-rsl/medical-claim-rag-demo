"""Pydantic models: the claim that goes in, and the decision that comes out.

Pydantic checks the type of every field when an object is created, so a malformed claim fails immediately.
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """What the decision step is allowed to see: the 18 claim fields, and no answers."""

    claim_id: str
    claimant_name: str
    age: int
    policy_start_date: date
    sum_insured: float
    plan: Literal["A", "B", "C"]
    voluntary_deductible: float  # 0 if the claimant chose none
    amount_claimed_ytd: float  # year to date: what has already been claimed this policy year
    pre_existing_declared: list[str]
    claim_type: Literal["inpatient", "outpatient", "daycare", "maternity", "dental", "emergency"]
    diagnosis: str
    procedure: str
    treatment_notes: str
    treatment_date: date
    is_accident: bool
    hospital_in_network: bool
    pre_auth_obtained: bool
    amount_claimed: float


class EvalClaim(Claim):
    """A test claim: the 18 claim fields plus the answer key.

    Only used to load the test file and to compare results afterwards. It is never passed to decide().
    """

    expected_decision: Literal["approve", "partial", "reject", "needs_review"]
    expected_payable: float | None
    expected_clauses: list[str]
    rationale: str


class Decision(BaseModel):
    """The structured answer the LLM must return.

    The field order matters: "reasoning" comes first so the model writes out its thinking before it
    commits to a decision. Each description is sent to the model, so it works as an instruction.
    """

    reasoning: str = Field(
        description="Step-by-step reasoning from the provided clauses and facts. "
        "If a payable amount is given, show each calculation step."
    )
    citations: list[str] = Field(
        description="Individual clause IDs relied on, e.g. '10.2.1'. Only IDs present in the provided context."
    )
    decision: Literal["approve", "partial", "reject", "needs_review"] = Field(
        description="Final decision on the claim."
    )
    payable_amount: float | None = Field(
        description="Amount payable: equals the claimed amount for approve, 0 for reject, "
        "between 0 and the claimed amount for partial, null for needs_review."
    )
    confidence: float = Field(
        description="0 to 1. 0.9+ if a clause directly and unambiguously decides it; "
        "0.6-0.8 if the clause applies but needs interpretation; below 0.5 if the clauses do not clearly address the claim."
    )
