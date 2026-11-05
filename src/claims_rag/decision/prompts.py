"""The prompt sent to the LLM, and the helpers that turn a claim, facts and chunks into prompt text."""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from claims_rag.decision.facts import Facts
from claims_rag.schemas import Claim

# The instructions. They are general rules and never mention a specific test claim.
SYSTEM_PROMPT = """You adjudicate medical insurance claims against a policy. Use ONLY the policy clauses and \
computed facts provided below. Do not use outside insurance knowledge.

Rules:
1. Cite the individual clause IDs you actually relied on, including sub-clauses such as 10.2.1. Cite only IDs that \
appear in the provided clauses.
2. Apply the policy's own precedence rules (Section 2.5, and clauses that say "notwithstanding", "subject to" or \
that a specific clause prevails). Check exclusions AND their exceptions before rejecting a claim.
3. Check waiting periods using the computed facts. Do not do date arithmetic yourself.
4. For a payable amount, follow the policy's calculation order (clause 14.6) and show each step in your reasoning. \
Plan limits come from the Schedule chunk, for the claimant's plan.
5. If the provided clauses do not clearly cover or exclude the situation, or the policy is silent or ambiguous, \
answer needs_review. Do not guess.

Decisions: approve = the full claimed amount is payable. partial = some amount, less than claimed, is payable. \
reject = nothing is payable. needs_review = a human must decide, with no payable amount.

Confidence: 0.9 or more if a clause directly and unambiguously decides the claim; 0.6 to 0.8 if the clause applies \
but needs interpretation; below 0.5 if the clauses do not clearly address the claim."""

# The message for one claim. The {placeholders} are filled in for every claim.
HUMAN_PROMPT = """CLAIM
{claim}

COMPUTED FACTS (calculated by software, treat as correct)
{facts}

POLICY CLAUSES
{context}
{retry_note}
Decide this claim."""

# ChatPromptTemplate is a list of message templates with {placeholders}. (Plain Python: an f-string.)
DECISION_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


def format_claim(claim: Claim) -> str:
    """Write the claim as 'field: value' lines. The ID and name are left out: they play no part in the decision."""
    fields_left_out = {"claim_id", "claimant_name"}
    lines = []
    for field_name, value in claim.model_dump().items():
        if field_name not in fields_left_out:
            lines.append(f"{field_name}: {value}")
    return "\n".join(lines)


def format_facts(facts: Facts) -> str:
    """Write the computed facts as a bullet list."""
    return "\n".join("- " + statement for statement in facts.as_statements())


def format_chunks(chunks: list[Document]) -> str:
    """Write the retrieved chunks one after another, each labelled with its chunk ID.

    Inside each chunk, every clause ID stays at the start of its own line, so the LLM can cite it.
    """
    labelled_chunks = []
    for chunk in chunks:
        labelled_chunks.append("[chunk " + chunk.metadata["chunk_id"] + "]\n" + chunk.page_content)
    return "\n\n".join(labelled_chunks)


def format_retry_note(problems: list[str]) -> str:
    """On a retry, tell the LLM what was wrong with its previous answer. On the first try there is no note."""
    if not problems:
        return ""
    return "\nYour previous answer was rejected: " + "; ".join(problems) + ". Correct these problems.\n"
