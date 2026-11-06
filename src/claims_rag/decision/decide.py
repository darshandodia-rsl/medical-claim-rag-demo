"""Decide a claim: retrieve the clauses, ask the LLM, check its answer, and send doubtful cases to a human."""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from groq import BadRequestError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from claims_rag.config import DECISION_LOG_PATH, LLM_MODEL
from claims_rag.decision.calculator import calculate
from claims_rag.decision.facts import Facts, compute_facts
from claims_rag.decision.prompts import DECISION_PROMPT, format_chunks, format_claim, format_facts, format_retry_note
from claims_rag.retrieval.retrieve import RetrievalResult, retrieve
from claims_rag.schemas import Claim, Decision

CONFIDENCE_FLOOR = 0.5  # below this confidence, a human reviews the claim
MAX_TOOL_ROUNDS = 5  # the LLM may ask for the calculator at most this many times in a row
# The final call has no tools attached. Sent as plain text, the calculator results cannot look like a tool
# conversation the model wants to continue (Groq answers with a 400 error if it tries to call a tool anyway).
FINAL_ANSWER_NOTE = "No more tool calls are available. Give your final decision now."
MAX_ATTEMPTS = 2  # the first try plus one retry
MONEY_TOLERANCE = 0.01  # amounts closer than this (one cent) count as equal


@dataclass
class DecisionResult:
    decision: Decision  # the final decision, after validation and routing
    llm_decision: Decision | None  # what the LLM said before routing (None if it never gave a usable answer)
    routed_by_confidence: bool  # True if low confidence changed the decision to needs_review
    retrieval: RetrievalResult
    facts: Facts
    attempts: list[dict] = field(default_factory=list)  # one record per LLM call, for the log


class ToolLoopChain:
    """Prompt, then a short calculator loop, then the final structured Decision.

    invoke() takes the prompt inputs and returns {'raw', 'parsed', 'parsing_error', 'tool_calls'}: the same shape
    as the plain chain plus a record of every calculator call. The loop is plain Python: ask the model, run the
    tools it asked for, send the results back, repeat until it stops asking (or MAX_TOOL_ROUNDS is reached).
    """

    def __init__(self, tool_llm, structured_llm):
        self.tool_llm = tool_llm  # the LLM with the calculator attached
        self.structured_llm = structured_llm  # the LLM that must answer with a Decision

    def invoke(self, prompt_inputs: dict) -> dict:
        prompt_messages = DECISION_PROMPT.format_messages(**prompt_inputs)
        messages = list(prompt_messages)  # the tool loop adds to this copy
        tool_calls = []

        for _ in range(MAX_TOOL_ROUNDS):
            reply = self.tool_llm.invoke(messages)
            if not reply.tool_calls:
                break
            messages.append(reply)
            for call in reply.tool_calls:
                result = calculate.invoke(call["args"])
                tool_calls.append({"expression": call["args"].get("expression"), "result": result})
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        # The final call starts again from the prompt and adds the calculator results as plain text.
        results_text = "".join(f"{call['expression']} = {call['result']}\n" for call in tool_calls)
        note = ("Calculator results (exact, computed by software):\n" + results_text if tool_calls else "") + FINAL_ANSWER_NOTE

        try:
            response = self.structured_llm.invoke(prompt_messages + [HumanMessage(content=note)])
        except BadRequestError as error:
            if "tool_use_failed" not in str(error):
                raise
            # The model tried to call a tool anyway. Report it as an unparseable answer so the normal retry applies.
            response = {"raw": AIMessage(content=str(error)), "parsed": None, "parsing_error": error}
        return {**response, "tool_calls": tool_calls}


def build_llm_chain():
    """Build the chain. Calling it returns {'raw', 'parsed', 'parsing_error', 'tool_calls'}.

    with_structured_output tells the LLM to answer in JSON that matches the Decision schema. 'parsed' is the
    Decision object, 'raw' is the model's message as sent, and 'parsing_error' is set if the JSON did not fit.
    (Plain Python: put the schema in the prompt, call the model, json.loads the reply, then Decision(**data).)
    bind_tools tells the model the calculator exists. (Plain Python: add the tool's JSON description to the request.)
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    # method="json_schema" makes Groq enforce the schema. The default (function calling) does not.
    llm_with_structure = llm.with_structured_output(Decision, method="json_schema", include_raw=True)
    llm_with_calculator = llm.bind_tools([calculate])

    return ToolLoopChain(llm_with_calculator, llm_with_structure)


def check_payable_amount(decision: Decision, amount_claimed: float) -> list[str]:
    """Check that the payable amount makes sense for the decision. Returns a list of problems."""
    payable = decision.payable_amount

    if decision.decision == "approve":
        # Approve means the whole claimed amount is paid.
        if payable is None or abs(payable - amount_claimed) > MONEY_TOLERANCE:
            return [f"approve requires payable_amount == amount claimed ({amount_claimed}), got {payable}"]
    elif decision.decision == "reject":
        # Reject means nothing is paid.
        if payable is None or abs(payable) > MONEY_TOLERANCE:
            return [f"reject requires payable_amount == 0, got {payable}"]
    elif decision.decision == "partial":
        # Partial means something is paid, but less than the claimed amount.
        if payable is None or not (0 < payable < amount_claimed):
            return [f"partial requires 0 < payable_amount < {amount_claimed}, got {payable}"]
    elif decision.decision == "needs_review":
        # A human decides, so there must be no amount yet.
        if payable is not None:
            return [f"needs_review requires payable_amount to be null, got {payable}"]

    return []


def check_decision(decision: Decision, claim: Claim, context_clause_ids: set[str]) -> list[str]:
    """Check the LLM's answer in plain Python. Returns a list of problems (an empty list means it is fine)."""
    problems = []

    # Every cited clause must be one the LLM was actually shown.
    unknown_citations = sorted(set(decision.citations) - context_clause_ids)
    if unknown_citations:
        problems.append(f"citations not in the provided clauses: {unknown_citations}")

    if not 0 <= decision.confidence <= 1:
        problems.append(f"confidence {decision.confidence} is outside 0 to 1")

    problems.extend(check_payable_amount(decision, claim.amount_claimed))
    return problems


def route_low_confidence(decision: Decision) -> tuple[Decision, bool]:
    """Send an unsure answer to a human review.

    Returns the decision to use, and True if it was changed. A decision that already says needs_review is not
    changed, so the flag is only True when low confidence actually overruled the LLM.
    """
    is_unsure = decision.confidence < CONFIDENCE_FLOOR
    if is_unsure and decision.decision != "needs_review":
        reviewed = decision.model_copy(update={"decision": "needs_review", "payable_amount": None})
        return reviewed, True
    return decision, False


def raw_message_text(raw_message) -> str:
    """The text of the model's raw reply. Some models return a list of parts instead of a plain string."""
    if isinstance(raw_message.content, str):
        return raw_message.content
    return json.dumps(raw_message.content, default=str)


def ask_llm_until_valid(chain, prompt_inputs: dict, claim: Claim, context_clause_ids: set[str]):
    """Call the LLM, check its answer, and retry once with an error note if the answer was not valid.

    Returns three things: the last Decision the LLM gave (None if it never gave a parseable one),
    one record per attempt (for the log), and the problems that remain (empty if the answer is valid).
    """
    attempts = []
    problems: list[str] = []
    llm_decision = None

    for _ in range(MAX_ATTEMPTS):
        # On a retry, the problems from the previous attempt are added to the prompt.
        response = chain.invoke({**prompt_inputs, "retry_note": format_retry_note(problems)})
        llm_decision = response.get("parsed")
        parsing_error = response.get("parsing_error")

        if llm_decision is None:
            # The reply could not be turned into a Decision at all.
            problems = [f"output could not be parsed: {parsing_error}"]
        else:
            problems = check_decision(llm_decision, claim, context_clause_ids)

        attempts.append(
            {
                "tool_calls": response.get("tool_calls", []),
                "raw_output": raw_message_text(response["raw"]),
                "parsed": llm_decision.model_dump() if llm_decision else None,
                "parsing_error": str(parsing_error) if parsing_error else None,
                "validation_errors": problems,
            }
        )

        if not problems:
            break  # a valid answer, no need to retry

    return llm_decision, attempts, problems


def decide(claim: Claim, vector_store=None, chain=None) -> DecisionResult:
    """Decide one claim from start to finish.

    Accepts a plain Claim only. An EvalClaim carries the answer key, so it is refused.
    The vector_store and chain arguments exist so tests can pass in fakes.
    """
    if type(claim) is not Claim:
        raise TypeError(f"decide() takes a Claim, got {type(claim).__name__}")
    if chain is None:
        chain = build_llm_chain()

    # Gather what the LLM needs: the relevant clauses, and the facts computed in Python.
    retrieval = retrieve(claim, vector_store)
    facts = compute_facts(claim)
    prompt_inputs = {
        "claim": format_claim(claim),
        "facts": format_facts(facts),
        "context": format_chunks(retrieval.chunks),
    }

    llm_decision, attempts, problems = ask_llm_until_valid(chain, prompt_inputs, claim, retrieval.context_clause_ids)

    if problems:
        # Still invalid after the retry: do not guess, hand the claim to a human and record why.
        final_decision = Decision(
            reasoning="Could not produce a valid decision: " + "; ".join(problems),
            citations=[],
            decision="needs_review",
            payable_amount=None,
            confidence=0.0,
        )
        routed_by_confidence = False
    else:
        final_decision, routed_by_confidence = route_low_confidence(llm_decision)

    result = DecisionResult(final_decision, llm_decision, routed_by_confidence, retrieval, facts, attempts)
    write_decision_log(claim, result)
    return result


def write_decision_log(claim: Claim, result: DecisionResult) -> None:
    """Append one line to the log file with everything needed to debug this run.

    That is the retrieved clauses, the raw LLM output of each attempt, the validation problems and the
    routing. It is stored on this computer only.
    """
    retrieved_chunks = []
    for chunk in result.retrieval.chunks:
        retrieved_chunks.append({"chunk_id": chunk.metadata["chunk_id"], "text": chunk.page_content})

    log_entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "claim_id": claim.claim_id,
        "model": LLM_MODEL,
        "query": result.retrieval.query,
        "retrieved": retrieved_chunks,
        "facts": result.facts.as_statements(),
        "attempts": result.attempts,
        "llm_decision": result.llm_decision.model_dump() if result.llm_decision else None,
        "routed_by_confidence": result.routed_by_confidence,
        "final_decision": result.decision.model_dump(),
    }

    DECISION_LOG_PATH.parent.mkdir(exist_ok=True)
    with DECISION_LOG_PATH.open("a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")
