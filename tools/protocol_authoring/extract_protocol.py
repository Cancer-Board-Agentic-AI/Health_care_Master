"""
Protocol authoring tool: LLM-assisted extraction of clinical protocol
content from a specialty-specific guideline excerpt (PDF or text).

This is an AUTHORING TOOL. It must never be imported by app/domain,
app/application, or app/infrastructure at runtime. It does not construct
AgentProtocol or SharedProtocol instances — it produces an intermediate,
human-reviewable JSON staging file. A human decides, per item, whether
`suggested_scope` routing is correct before any content is turned into a
real SharedProtocol or AgentProtocol JSON file.

Usage:
    python extract_protocol.py \
      --input path/to/guideline.pdf \
      --specialty pathology \
      --cancer-type breast_cancer \
      --population-context india \
      --output tools/protocol_authoring/drafts/pathology_extraction.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Specialty-aware extraction guidance (a hint list, NOT clinical logic)
# ---------------------------------------------------------------------------

SPECIALTY_HINTS: dict[str, str] = {
    "pathology": (
        "histology, grade, receptor status (ER/PR/HER2), margins, "
        "lymphovascular invasion, biomarker/molecular findings"
    ),
    "radiology": (
        "tumor size/location, nodal status, distant metastasis, "
        "imaging-based staging"
    ),
    "surgery": (
        "resectability, operability, surgical options, neoadjuvant timing"
    ),
    "medical_oncology": (
        "systemic therapy options, sequencing, contraindications"
    ),
    "radiation": (
        "indication, intent (curative/palliative), target volume, fractionation"
    ),
    "supportive": (
        "ECOG, symptom burden, nutrition, psychosocial needs"
    ),
}

VALID_SPECIALTIES = tuple(SPECIALTY_HINTS.keys())

SHARED_HINT = (
    "Regardless of specialty, always flag TNM staging categories, biomarker "
    "definitions/value sets, and generic validation/safety/confidence-scoring "
    "concepts as likely 'shared_definitional' or 'shared_behavioral' "
    "candidates, since these are explicitly cross-specialty concerns."
)

CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 500
MAX_RETRIES_PER_KEY = 3
DEFAULT_MODEL = "llama-3.3-70b-versatile"
RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "429", "quota", "too many requests")


# ---------------------------------------------------------------------------
# Multi-key API pool
#
# Groq's free/dev tier has per-key rate limits (requests/min, tokens/min,
# requests/day). To keep long extraction runs from breaking on a single
# key's limit, we accept as many keys as the user provides and rotate
# across them whenever a call looks rate-limited, before falling back to
# exponential backoff on the same key for transient errors (timeouts etc).
# ---------------------------------------------------------------------------

def load_api_keys() -> list[str]:
    """
    Collects Groq API keys from the environment. Supports any of:
      - GROQ_API_KEYS=key1,key2,key3      (comma-separated, any count)
      - GROQ_API_KEY=key1                 (single key, backward compatible)
      - GROQ_API_KEY_1=key1, GROQ_API_KEY_2=key2, ...  (numbered, any count)
    All forms may be combined; duplicates are dropped, order preserved.
    """
    keys: list[str] = []
    seen: set[str] = set()

    def add(key: str | None) -> None:
        if key and key.strip() and key.strip() not in seen:
            seen.add(key.strip())
            keys.append(key.strip())

    add(os.environ.get("GROQ_API_KEY"))

    for raw in (os.environ.get("GROQ_API_KEYS") or "").split(","):
        add(raw)

    i = 1
    while True:
        val = os.environ.get(f"GROQ_API_KEY_{i}")
        if val is None:
            break
        add(val)
        i += 1

    return keys


def is_rate_limit_error(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if status == 429:
        return True
    text = str(err).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


class GroqKeyPool:
    """Round-robins across multiple Groq clients, skipping keys that are
    currently rate-limited and retrying with backoff on transient errors."""

    def __init__(self, api_keys: list[str]):
        from groq import Groq

        if not api_keys:
            raise ValueError("No Groq API keys provided")
        self._clients = [Groq(api_key=k) for k in api_keys]
        self._index = 0

    @property
    def size(self) -> int:
        return len(self._clients)

    def next_client(self):
        client = self._clients[self._index]
        self._index = (self._index + 1) % len(self._clients)
        return client


# ---------------------------------------------------------------------------
# Intermediate output schema (staging only — NOT AgentProtocol/SharedProtocol)
# ---------------------------------------------------------------------------

class ExtractedDefinition(BaseModel):
    term: str
    definition: str
    valid_values: list[str] | None = None
    source_excerpt: str
    suggested_scope: str
    scope_rationale: str


class ExtractedRule(BaseModel):
    description: str
    condition_hint: str
    action_hint: str
    source_excerpt: str
    suggested_scope: str
    scope_rationale: str


class ExtractionMetadata(BaseModel):
    source_file: str
    specialty_hint: str
    cancer_type: str
    population_context: str | None
    model_used: str
    extracted_at: str


class RejectedItem(BaseModel):
    item: dict[str, Any]
    reason: str


class ExtractionResult(BaseModel):
    extraction_metadata: ExtractionMetadata
    extracted_definitions: list[ExtractedDefinition] = []
    extracted_rules: list[ExtractedRule] = []
    extracted_required_inputs: list[str] = []
    rejected_items: list[RejectedItem] = []


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_input_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(
                "ERROR: pypdf is required to read PDF input. Install with "
                "`pip install pypdf`.",
                file=sys.stderr,
            )
            sys.exit(1)
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        print(f"ERROR: unsupported input file type: {path.suffix}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(chunk: str, specialty: str, cancer_type: str, population_context: str | None) -> str:
    specialty_hint = SPECIALTY_HINTS[specialty]
    population_line = (
        f"Population/regional context: {population_context}\n"
        if population_context
        else ""
    )
    return f"""You are extracting structured clinical protocol content from a guideline
excerpt for the "{specialty}" specialty, for cancer type "{cancer_type}".
{population_line}
STRICT RULES:
1. Extract ONLY what is explicitly stated in the text below. Never infer,
   estimate, or supplement using general/pretrained medical knowledge.
2. Every extracted definition or rule MUST include a `source_excerpt` field.
   VERBATIM EXCERPT RULES (non-negotiable):
   - The source_excerpt field must be an EXACT, CHARACTER-FOR-CHARACTER COPY
     of a contiguous span of the input text below.
   - Do NOT paraphrase, summarize, reword, or "clean up" the text.
   - Do NOT merge two separate sentences into one unless they are truly
     contiguous in the source (adjacent, nothing removed between them).
   - Do NOT fix, normalize, or modernize spelling, spacing, hyphenation, or
     punctuation (e.g. if the source says "ER-PR" or "ERPR" or "ER/PR", copy
     exactly what appears — do not standardize it).
   - If you need to convey information that spans a long or non-contiguous
     passage, either (a) quote the single shortest CONTIGUOUS span that
     supports the claim, or (b) split it into multiple separate extracted
     items, each with its own short verbatim excerpt — never stitch
     non-adjacent sentences together into one artificial excerpt.
   - The `definition` / `description` / `action_hint` fields are where you
     may paraphrase or explain in your own words. The `source_excerpt` field
     must never contain your own words — only the source's, copied exactly.
   - If you cannot find a supporting verbatim contiguous excerpt, do not
     include the item.

   Example (generic, non-medical, illustrating the rule):
   BAD  (paraphrased/compressed — reject):
     {{"definition": "Values above 1% are considered positive",
       "source_excerpt": "Any value over 1% counts as a positive result"}}
   GOOD (copied character-for-character from the source — accept):
     {{"definition": "Values above 1% are considered positive",
       "source_excerpt": "results exceeding 1% shall be classified as positive"}}
   The GOOD example's source_excerpt is exactly what appeared in the source
   text, not a cleaned-up restatement of it. Match that standard.
3. For each item, set `suggested_scope` to one of:
   - "shared_definitional": a closed vocabulary / structural definition
     (e.g. a TNM category, a biomarker value set) that plausibly applies
     beyond this one specialty.
   - "shared_behavioral": a cross-cutting rule (validation, safety,
     confidence, missing-data handling) rather than specialty-specific
     clinical judgment.
   - "specialty_specific": reasoning that only makes sense within this one
     specialty's scope of practice.
   Include a one-sentence `scope_rationale` explaining why you chose that tag.
4. {SHARED_HINT}
5. This specialty typically involves concepts like: {specialty_hint}. Use this
   only as a hint for what to look for — do not force-fit unrelated content.

Return a single JSON object with exactly these top-level keys:
{{
  "definitions": [
    {{"term": "...", "definition": "...", "valid_values": ["..."] or null,
      "source_excerpt": "...", "suggested_scope": "...", "scope_rationale": "..."}}
  ],
  "rules": [
    {{"description": "...", "condition_hint": "...", "action_hint": "...",
      "source_excerpt": "...", "suggested_scope": "...", "scope_rationale": "..."}}
  ],
  "required_inputs": ["..."]
}}

If nothing extractable is found in this excerpt, return empty lists for all keys.

INPUT TEXT:
---
{chunk}
---
"""


# ---------------------------------------------------------------------------
# Self-check pass: a second, lightweight Groq call that reviews the
# model's own draft excerpts against the source chunk and corrects or
# drops any that are not exact verbatim substrings, before the item ever
# reaches the Python-level substring check.
# ---------------------------------------------------------------------------

def build_self_check_prompt(chunk: str, definitions: list[dict[str, Any]], rules: list[dict[str, Any]]) -> str:
    items_for_review = {
        "definitions": [
            {"index": i, "source_excerpt": d.get("source_excerpt", "")}
            for i, d in enumerate(definitions)
        ],
        "rules": [
            {"index": i, "source_excerpt": r.get("source_excerpt", "")}
            for i, r in enumerate(rules)
        ],
    }
    return f"""You are verifying draft extractions against their source text.

For each item below, verify its `source_excerpt` is an EXACT, character-for-
character substring of the SOURCE TEXT provided below (ignoring only leading/
trailing whitespace and run-length of internal whitespace). If it is not an
exact substring:
  - Try to correct it to the exact verbatim substring from SOURCE TEXT that
    best supports the same claim, OR
  - If no such exact supporting span exists anywhere in SOURCE TEXT, mark the
    item for removal.
Do NOT invent new source_excerpt text. Do NOT paraphrase. Only trim or
correct toward text that verbatim appears in SOURCE TEXT, or remove the item.
Items whose source_excerpt is already an exact substring should be returned
unchanged.

Return a single JSON object with exactly these keys, one entry per input
item (by index), each entry either corrected or marked removed:
{{
  "definitions": [
    {{"index": 0, "source_excerpt": "...", "action": "keep" | "corrected" | "remove"}}
  ],
  "rules": [
    {{"index": 0, "source_excerpt": "...", "action": "keep" | "corrected" | "remove"}}
  ]
}}

ITEMS TO REVIEW:
{json.dumps(items_for_review, indent=2)}

SOURCE TEXT:
---
{chunk}
---
"""


def self_check_chunk(
    pool: "GroqKeyPool",
    model: str,
    chunk: str,
    definitions: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not definitions and not rules:
        return definitions, rules

    prompt = build_self_check_prompt(chunk, definitions, rules)
    try:
        review = call_groq(pool, model, prompt)
    except RuntimeError as e:
        print(f"WARNING: self-check pass failed, proceeding without it: {e}", file=sys.stderr)
        return definitions, rules

    def apply_review(items: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_index = {r.get("index"): r for r in reviewed if isinstance(r, dict)}
        result = []
        for i, item in enumerate(items):
            r = by_index.get(i)
            if r is None:
                result.append(item)
                continue
            if r.get("action") == "remove":
                continue
            corrected = dict(item)
            if r.get("source_excerpt"):
                corrected["source_excerpt"] = r["source_excerpt"]
            result.append(corrected)
        return result

    corrected_definitions = apply_review(definitions, review.get("definitions", []) or [])
    corrected_rules = apply_review(rules, review.get("rules", []) or [])
    return corrected_definitions, corrected_rules


# ---------------------------------------------------------------------------
# Groq API call with retry
# ---------------------------------------------------------------------------

def call_groq(pool: GroqKeyPool, model: str, prompt: str) -> dict[str, Any]:
    """
    Tries every key in the pool, up to MAX_RETRIES_PER_KEY attempts each.
    Rate-limit errors rotate to the next key immediately (no wasted wait);
    other transient errors (timeouts, 5xx) retry the same key with
    exponential backoff. Only gives up once every key has exhausted its
    retries, so a single key's limit never breaks the run as long as at
    least one key still has headroom.
    """
    last_error: Exception | None = None
    total_keys = pool.size

    for key_round in range(total_keys):
        client = pool.next_client()
        for attempt in range(1, MAX_RETRIES_PER_KEY + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    print(
                        f"WARNING: model returned non-JSON output (attempt {attempt}): {e}",
                        file=sys.stderr,
                    )
                    last_error = e
                    continue
            except Exception as e:  # noqa: BLE001 - Groq SDK raises varied error types
                last_error = e
                if is_rate_limit_error(e):
                    print(
                        f"WARNING: key {key_round + 1}/{total_keys} rate-limited: {e}. "
                        "Rotating to next key.",
                        file=sys.stderr,
                    )
                    break  # stop retrying this key, move to next one immediately
                if attempt < MAX_RETRIES_PER_KEY:
                    wait = 2 ** attempt
                    print(
                        f"WARNING: Groq API call failed (key {key_round + 1}/{total_keys}, "
                        f"attempt {attempt}/{MAX_RETRIES_PER_KEY}): {e}. Retrying in {wait}s...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"WARNING: key {key_round + 1}/{total_keys} exhausted retries: {e}. "
                        "Rotating to next key.",
                        file=sys.stderr,
                    )

    raise RuntimeError(
        f"Groq extraction failed after trying all {total_keys} key(s): {last_error}"
    )


# ---------------------------------------------------------------------------
# Merge / dedupe across chunks
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def merge_chunk_results(chunk_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    seen_def_excerpts: set[str] = set()
    seen_rule_excerpts: set[str] = set()
    seen_inputs: set[str] = set()

    definitions: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    required_inputs: list[str] = []

    for output in chunk_outputs:
        for d in output.get("definitions", []) or []:
            key = _normalize(d.get("source_excerpt", ""))
            if key and key not in seen_def_excerpts:
                seen_def_excerpts.add(key)
                definitions.append(d)
        for r in output.get("rules", []) or []:
            key = _normalize(r.get("source_excerpt", ""))
            if key and key not in seen_rule_excerpts:
                seen_rule_excerpts.add(key)
                rules.append(r)
        for i in output.get("required_inputs", []) or []:
            key = _normalize(i)
            if key and key not in seen_inputs:
                seen_inputs.add(key)
                required_inputs.append(i)

    return {
        "definitions": definitions,
        "rules": rules,
        "required_inputs": required_inputs,
    }


# ---------------------------------------------------------------------------
# Source-excerpt validation (non-LLM)
# ---------------------------------------------------------------------------

def excerpt_found_in_source(excerpt: str, source_text: str) -> bool:
    return _normalize(excerpt) in _normalize(source_text)


NEAR_MATCH_THRESHOLD = 0.85


def excerpt_similarity(excerpt: str, source_text: str) -> float:
    """
    Fuzzy similarity of `excerpt` to its best-matching local window in
    `source_text`, using difflib. Used only for rejection diagnostics —
    never to accept an item. A high score means the model likely
    paraphrased/reworded real source content; a low score means the
    excerpt has little in common with anything in the source (more
    likely fabricated).
    """
    norm_excerpt = _normalize(excerpt)
    norm_source = _normalize(source_text)
    if not norm_excerpt or not norm_source:
        return 0.0

    matcher = difflib.SequenceMatcher(None, norm_excerpt, norm_source, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_excerpt), 0, len(norm_source))
    if match.size == 0:
        return 0.0

    pad = max(0, len(norm_excerpt) - match.size) + 20
    window_start = max(0, match.b - pad)
    window_end = min(len(norm_source), match.b + match.size + pad)
    window = norm_source[window_start:window_end]

    return difflib.SequenceMatcher(None, norm_excerpt, window, autojunk=False).ratio()


def _rejection_reason(excerpt: str, source_text: str) -> str:
    if not excerpt:
        return "source_excerpt missing"
    similarity = excerpt_similarity(excerpt, source_text)
    if similarity >= NEAR_MATCH_THRESHOLD:
        return (
            f"near_match_likely_paraphrased (similarity={similarity:.2f}) — "
            "content is probably real but reworded by the model; verbatim "
            "excerpt not found in source"
        )
    return (
        f"source_excerpt not found in input text (similarity={similarity:.2f} "
        "— low similarity to anything in source, may be fabricated)"
    )


def validate_items(merged: dict[str, Any], source_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid_definitions = []
    valid_rules = []
    rejected: list[dict[str, Any]] = []

    for d in merged["definitions"]:
        excerpt = d.get("source_excerpt", "")
        if excerpt and excerpt_found_in_source(excerpt, source_text):
            valid_definitions.append(d)
        else:
            rejected.append({"item": d, "reason": _rejection_reason(excerpt, source_text)})

    for r in merged["rules"]:
        excerpt = r.get("source_excerpt", "")
        if excerpt and excerpt_found_in_source(excerpt, source_text):
            valid_rules.append(r)
        else:
            rejected.append({"item": r, "reason": _rejection_reason(excerpt, source_text)})

    validated = {
        "definitions": valid_definitions,
        "rules": valid_rules,
        "required_inputs": merged["required_inputs"],
    }
    return validated, rejected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured clinical protocol content from a guideline excerpt using Groq."
    )
    parser.add_argument("--input", required=True, help="Path to input PDF or .txt file")
    parser.add_argument(
        "--specialty",
        required=True,
        choices=VALID_SPECIALTIES,
        help="Specialty hint for extraction guidance",
    )
    parser.add_argument("--cancer-type", required=True, help="Cancer type, e.g. breast_cancer")
    parser.add_argument(
        "--population-context",
        default=None,
        help="Optional population/regional context, e.g. india",
    )
    parser.add_argument("--output", required=True, help="Path to write the draft JSON output")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model to use")
    args = parser.parse_args()

    load_dotenv()

    try:
        from groq import Groq  # noqa: F401 - import check before building the pool
    except ImportError:
        print(
            "ERROR: groq package is required. Install with `pip install groq`.",
            file=sys.stderr,
        )
        sys.exit(1)

    api_keys = load_api_keys()
    if not api_keys:
        print(
            "ERROR: no Groq API key found. Set GROQ_API_KEY, GROQ_API_KEYS "
            "(comma-separated), or GROQ_API_KEY_1/_2/... in a .env file "
            "(see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)

    pool = GroqKeyPool(api_keys)
    print(f"Loaded {pool.size} Groq API key(s) for rotation.", file=sys.stderr)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    source_text = load_input_text(input_path)
    if not source_text.strip():
        print("ERROR: input file produced no extractable text.", file=sys.stderr)
        sys.exit(1)

    chunks = chunk_text(source_text)

    chunk_outputs: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Extracting chunk {i}/{len(chunks)}...", file=sys.stderr)
        prompt = build_prompt(chunk, args.specialty, args.cancer_type, args.population_context)
        try:
            result = call_groq(pool, args.model, prompt)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Self-checking chunk {i}/{len(chunks)} excerpts...", file=sys.stderr)
        definitions, rules = self_check_chunk(
            pool, args.model, chunk, result.get("definitions", []) or [], result.get("rules", []) or []
        )
        result["definitions"] = definitions
        result["rules"] = rules
        chunk_outputs.append(result)

    merged = merge_chunk_results(chunk_outputs)
    validated, rejected = validate_items(merged, source_text)

    if not validated["definitions"] and not validated["rules"] and not validated["required_inputs"]:
        print(
            "ERROR: extraction produced zero definitions, rules, and required_inputs. "
            "This likely means the model found nothing usable in the input text. "
            "No output file written.",
            file=sys.stderr,
        )
        if rejected:
            print(
                f"({len(rejected)} item(s) were extracted but rejected for failing "
                "source_excerpt validation — see below)",
                file=sys.stderr,
            )
            for r in rejected:
                print(f"  - {r['reason']}: {r['item']}", file=sys.stderr)
        sys.exit(1)

    metadata = ExtractionMetadata(
        source_file=str(input_path),
        specialty_hint=args.specialty,
        cancer_type=args.cancer_type,
        population_context=args.population_context,
        model_used=args.model,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        extraction_result = ExtractionResult(
            extraction_metadata=metadata,
            extracted_definitions=validated["definitions"],
            extracted_rules=validated["rules"],
            extracted_required_inputs=validated["required_inputs"],
            rejected_items=rejected,
        )
    except ValidationError as e:
        print(f"ERROR: extracted content failed intermediate schema validation:\n{e}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        extraction_result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        f"Wrote draft extraction to {output_path} "
        f"({len(validated['definitions'])} definitions, "
        f"{len(validated['rules'])} rules, "
        f"{len(validated['required_inputs'])} required_inputs, "
        f"{len(rejected)} rejected items). "
        "This is a REVIEW-STAGING file — a human must confirm suggested_scope "
        "before any content becomes a SharedProtocol or AgentProtocol JSON file.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
