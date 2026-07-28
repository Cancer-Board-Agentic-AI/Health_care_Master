# Protocol Authoring: Extraction Tool

`extract_protocol.py` is an **authoring-time** tool that uses the Groq API
to pull structured clinical protocol content out of a specialty-specific
guideline excerpt (PDF or text).

**This tool must never be imported by `app/domain`, `app/application`, or
`app/infrastructure` at runtime.** It lives entirely under
`tools/protocol_authoring/`.

## What this produces — and what it does NOT produce

This script writes **one intermediate, human-reviewable JSON draft file**.
It does **not** construct `AgentProtocol` or `SharedProtocol` instances, and
it does **not** decide where content ultimately belongs.

Every extracted item carries a `suggested_scope` tag:

- `shared_definitional` — a closed vocabulary / structural definition
  (e.g. a TNM category, a biomarker value set) that plausibly applies
  beyond the one specialty being processed.
- `shared_behavioral` — a cross-cutting rule (validation, safety,
  confidence, missing-data handling) rather than specialty-specific
  clinical judgment.
- `specialty_specific` — reasoning that only makes sense within this one
  specialty's scope of practice.

along with a one-sentence `scope_rationale`.

**These tags are suggestions for human review, not a final classification.**
Misclassifying a concept as specialty-specific when it is actually shared
(or vice versa) silently breaks the single-source-of-truth guarantee the
two-tier protocol architecture depends on. A human must review every item
and manually route it into a `SharedProtocol` or `AgentProtocol` JSON file
— this tool intentionally never writes to `shared_protocols/` or to a
specialty protocol file directly.

## Extraction guarantees

- The model extracts only what is **explicitly stated** in the input text.
  Nothing is inferred, estimated, or filled from general/pretrained
  knowledge.
- Every extracted definition or rule includes a `source_excerpt` — the
  verbatim sentence(s) from the input text that justify it.
- After extraction, a **plain-Python, non-LLM validation step** checks that
  every `source_excerpt` actually appears (whitespace-normalized) as a
  substring of the original input text. Items that fail this check are
  moved to `rejected_items` with a reason — never silently dropped, never
  included in the main output.
- If extraction yields zero definitions, zero rules, and zero
  required_inputs, the script fails loudly and does not write an output
  file.

## Install

```bash
pip install groq python-dotenv pypdf pydantic
```

(See `requirements.txt` at the repo root if present — these packages have
been added there.)

Copy `.env.example` to `.env` and set your key(s):

```bash
cp tools/protocol_authoring/.env.example tools/protocol_authoring/.env
# then edit .env and set your Groq API key(s)
```

`extract_protocol.py` loads `.env` via `python-dotenv` from the current
working directory by default — run it from the repo root, or place a
`.env` there, or export the variables directly in your shell.

### Multiple API keys (rate-limit resilience)

Groq's per-key rate limits (requests/min, tokens/min, requests/day) can be
hit during a long extraction run. The script accepts **as many keys as you
want** and rotates across them automatically. Any of these forms work, and
can be combined:

```bash
# Single key
GROQ_API_KEY=key_one

# Comma-separated list (any number of keys)
GROQ_API_KEYS=key_one,key_two,key_three

# Numbered keys (any count, sequential from 1)
GROQ_API_KEY_1=key_one
GROQ_API_KEY_2=key_two
GROQ_API_KEY_3=key_three
```

Behavior:
- Keys are used round-robin, one call at a time.
- If a call looks rate-limited (HTTP 429, or an error message mentioning
  "rate limit" / "quota" / "too many requests"), the script rotates to the
  next key **immediately** instead of waiting or burning retries.
- Other transient errors (timeouts, 5xx) retry up to 3 times on the same
  key with exponential backoff before moving to the next key.
- The run only fails once **every** key has been tried and exhausted —
  so as long as at least one key still has headroom, a single key's limit
  will not break extraction.

## Usage

```bash
python tools/protocol_authoring/extract_protocol.py \
  --input path/to/pathology_guideline_excerpt.pdf \
  --specialty pathology \
  --cancer-type breast_cancer \
  --population-context india \
  --output tools/protocol_authoring/drafts/pathology_extraction.json
```

### CLI arguments

| Flag | Required | Description |
|---|---|---|
| `--input` | yes | Path to a `.pdf` or `.txt` guideline excerpt (pre-trimmed to one specialty) |
| `--specialty` | yes | One of: `pathology`, `radiology`, `surgery`, `medical_oncology`, `radiation`, `supportive` — steers extraction guidance only, no clinical logic |
| `--cancer-type` | yes | e.g. `breast_cancer` — passed through as metadata |
| `--population-context` | no | e.g. `india` — passed through as metadata |
| `--output` | yes | Path to write the draft JSON |
| `--model` | no | Groq model, default `llama-3.3-70b-versatile` |

## Output shape

```jsonc
{
  "extraction_metadata": {
    "source_file": "...",
    "specialty_hint": "pathology",
    "cancer_type": "breast_cancer",
    "population_context": "india",
    "model_used": "llama-3.3-70b-versatile",
    "extracted_at": "2026-07-27T12:00:00+00:00"
  },
  "extracted_definitions": [
    {
      "term": "...", "definition": "...", "valid_values": ["..."] ,
      "source_excerpt": "...", "suggested_scope": "shared_definitional",
      "scope_rationale": "..."
    }
  ],
  "extracted_rules": [
    {
      "description": "...", "condition_hint": "...", "action_hint": "...",
      "source_excerpt": "...",
      "suggested_scope": "shared_behavioral",
      "scope_rationale": "..."
    }
  ],
  "extracted_required_inputs": ["..."],
  "rejected_items": [
    {"item": {"...": "..."}, "reason": "source_excerpt not found in input text"}
  ]
}
```

## Chunking

If the input text exceeds ~12,000 characters, it is split into overlapping
chunks (500-char overlap) and extracted per-chunk. Results are merged with
deduplication by normalized `source_excerpt` text.

## Next step (manual)

Open the draft JSON, review each item's `suggested_scope` and
`scope_rationale`, and manually author or update the corresponding
`SharedProtocol` (`kind=DEFINITIONAL`/`BEHAVIORAL`) or `AgentProtocol`
JSON file per the shapes defined in your domain schema. This tool does not
perform that conversion.
