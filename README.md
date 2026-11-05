# Medical Claims RAG

A small RAG system that decides medical insurance claims against a policy document and cites the clauses it used to arrive at that decision. All data is fictional (Meridian Health Assurance) generated via AI.

## Inputs and outputs

- **Policy:** The policy wording is a single markdown file (`data/policy/meridian_care_plus_policy.md`) that is read during ingestion.
- **Input:** Each claim is a structured JSON object with 18 fields (see the `Claim` schema in `src/claims_rag/schemas.py`). Most fields are typed values such as dates, amounts, the plan and the claim type. The free-text fields are `diagnosis`, `procedure` and `treatment_notes`, and `treatment_notes` is the long narrative written in natural language.
- **Output:** The decision is also structured JSON (see the `Decision` schema): `reasoning`, `citations`, `decision`, `payable_amount` and `confidence`.


## How it works

```
Claim ─┬─ computed facts (dates, age 60+)
       └─ hybrid search (embeddings + BM25) ─► top 15 chunks + Schedule 16.1
                         │
        facts + chunks ──► LLM (structured output) ──► validation ──► Decision
```

- **Chunking:** One chunk per policy clause and sub-clauses are merged into their parent (155 clauses, 133 chunks).
- **Retrieval:** A hybrid search approach with Embedding search and BM25 keyword search, merged with Reciprocal Rank Fusion, weighted 70% embedding and 30% keyword.
- **Facts:** Date arithmetic is done in Python so it stays deterministic as opposed to LLM.
- **LLM:** An open source model `openai/gpt-oss-120b`, with structured JSON output, temperature 0 used via Groq API.
- **Validation:** Validate each LLM output by checking that it did not make up any citations of its own, along with the payable amount. If it did hallcinate any of those, do a retry once.


## Run it

Python 3.11+.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                 # set GROQ_API_KEY and LLM_MODEL
python -m claims_rag.cli index
python -m claims_rag.cli retrieve --claim C004       # only retreives the chunks
python -m claims_rag.cli decide --claim C004         # runs full RAG pipline
python -m claims_rag.retrieval.eval_retrieval        # evaluates retreival metrics against test data
```
