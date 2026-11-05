"""Turn the policy markdown into chunks: one chunk per clause, with sub-clauses merged into their parent.

Each chunk is a LangChain Document (text plus a metadata dictionary). The clause-aware parsing is written
by hand on purpose: clause boundaries are the whole point, so a generic text splitter would not do.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from claims_rag.config import POLICY_PATH

# A clause starts at the beginning of a line with a number like 4.4 or 10.2.1 followed by a space.
CLAUSE_LINE_PATTERN = re.compile(r"^(\d+(?:\.\d+)+)\s")

# A section heading looks like "## 10. PERMANENT EXCLUSIONS".
SECTION_HEADING_PATTERN = re.compile(r"^##\s+(\d+)\.\s+(.+)$")


@dataclass
class Clause:
    """One numbered clause of the policy, such as 10.2.1, with the lines of text that belong to it."""

    clause_id: str
    section_number: int
    section_title: str
    lines: list[str]


def chunk_type_for_section(section_number: int) -> str:
    """Say what kind of rule a section contains. Stored in the metadata; not used for filtering in v1."""
    if section_number == 1:
        return "definition"
    if section_number == 2:
        return "scope"
    if 3 <= section_number <= 7:
        return "coverage"
    if section_number in (8, 9):
        return "waiting_period"
    if section_number == 10:
        return "exclusion"
    if section_number == 11:
        return "limit"
    if section_number in (12, 13):
        return "procedure"
    if section_number == 14:
        return "calculation"
    if section_number == 15:
        return "general"
    return "schedule"  # section 16


def parent_clause_id(clause_id: str) -> str:
    """Return the ID of the chunk a clause belongs to.

    A sub-clause with three parts (10.2.1) belongs to its parent with two parts (10.2).
    Any other clause (4.4, 10.2) is a chunk on its own.
    """
    parts = clause_id.split(".")
    if len(parts) > 2:
        return parts[0] + "." + parts[1]
    return clause_id


def split_into_clauses(policy_text: str) -> list[Clause]:
    """Split the policy text into individual clauses, in the order they appear.

    A clause's text runs from its ID line until the next clause ID or the next section heading.
    That way clauses that span several lines, and the Schedule tables, stay whole without special handling.
    """
    clauses: list[Clause] = []
    section_number = 0
    section_title = ""
    current_clause = None  # the clause we are collecting lines for right now

    for line in policy_text.splitlines():
        heading_match = SECTION_HEADING_PATTERN.match(line)
        if heading_match:
            section_number = int(heading_match.group(1))
            section_title = heading_match.group(2).strip().title()
            current_clause = None  # text right after a heading belongs to no clause
            continue

        clause_match = CLAUSE_LINE_PATTERN.match(line)
        if clause_match and section_number > 0:
            # A new clause starts on this line.
            current_clause = Clause(clause_match.group(1), section_number, section_title, [line.rstrip()])
            clauses.append(current_clause)
        elif current_clause is not None and line.strip() != "" and line.strip() != "---":
            # A continuation line (more text, or a table row) of the clause we are inside.
            current_clause.lines.append(line.rstrip())

    return clauses


def build_chunks(policy_path: Path = POLICY_PATH) -> list[Document]:
    """Read the policy and produce one Document per chunk.

    A clause with sub-clauses (10.2 with 10.2.1) becomes a single chunk, so a rule and its exception are
    always retrieved together. Every clause ID stays at the start of its own line inside the chunk text,
    so the LLM can cite the exact sub-clause. (Plain-Python alternative to Document: a dict with the same fields.)
    """
    clauses = split_into_clauses(policy_path.read_text())

    # Group the clauses by the chunk they belong to. A dict remembers insertion order, so chunks stay in policy order.
    clauses_by_chunk_id: dict[str, list[Clause]] = {}
    for clause in clauses:
        chunk_id = parent_clause_id(clause.clause_id)
        if chunk_id not in clauses_by_chunk_id:
            clauses_by_chunk_id[chunk_id] = []
        clauses_by_chunk_id[chunk_id].append(clause)

    chunks = []
    for chunk_id, chunk_clauses in clauses_by_chunk_id.items():
        # The chunk text is every line of every clause in the group, in order.
        chunk_lines = []
        for clause in chunk_clauses:
            chunk_lines.extend(clause.lines)

        first_clause = chunk_clauses[0]
        clause_ids = [clause.clause_id for clause in chunk_clauses]
        metadata = {
            "chunk_id": chunk_id,
            # Chroma metadata only allows single values (no lists), so the clause IDs are joined with commas.
            "clause_ids": ",".join(clause_ids),
            "section": first_clause.section_number,
            "section_title": first_clause.section_title,
            "type": chunk_type_for_section(first_clause.section_number),
        }
        chunks.append(Document(page_content="\n".join(chunk_lines), metadata=metadata))

    return chunks
