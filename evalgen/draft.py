"""Step 3a: draft the expected output of each selected case with an LLM.

The drafter sees the user message, the documentation the assistant had and
the assistant policy. It does NOT see the production response, so it cannot
copy a hallucinated number into the reference. Drafts are checked
deterministically (every number in a required fact must exist in the
documentation) and every draft goes to a human reviewer anyway.
"""

from __future__ import annotations

import time

from .assistant import POLICY, TOOL_DOCS
from .schema import ACTIONS, TOOLS, EvalItem, Expected, SchemaError
from .synth.kb import context_text
from .textutil import extract_numbers, parse_json_loose

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "tool": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": list(TOOLS)}, "arguments": {"type": "object"}},
                    "required": ["name", "arguments"],
                },
            ]
        },
        "required_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "reference_answer": {"type": "string"},
    },
    "required": ["action", "tool", "required_facts", "reference_answer"],
}

DRAFT_SYSTEM = f"""You write the expected output for one test case of an evaluation set.
The system under test is the Ledgerfox customer support assistant. It must follow this policy:

{POLICY}

{TOOL_DOCS}

Given the documentation the assistant could see and the user message, return JSON with:
- "action": what a correct assistant does: "answer", "tool_call" or "decline".
- "tool": for "tool_call", the tool name and its key argument (invoice_id for lookup_invoice, account_id for create_refund_request, reason for escalate_to_human); otherwise null.
- "required_facts": for "answer" only, 1 to 3 facts from the documentation that a correct reply must state to answer this specific question. Leave out related facts the user did not ask about. Each fact is a short phrase of at most 6 words that keeps the exact number or name, for example "$29 per month" or "up to 5 users". Use [] for "tool_call" and "decline".
- "reference_answer": one or two sentences a correct assistant would say.
Use only the documentation. Never add a fact it does not contain."""


def draft_messages(item: EvalItem, kb: dict) -> list[dict]:
    user = (
        f"Documentation:\n{context_text(item.context_doc_ids, kb)}\n\n"
        f"User message:\n{item.user_message}"
    )
    return [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}]


def check_draft(expected: Expected, item: EvalItem, kb: dict) -> list[str]:
    """Deterministic sanity checks on a draft. Returns warnings (empty = clean)."""
    warnings = []
    allowed = extract_numbers(context_text(item.context_doc_ids, kb)) | extract_numbers(item.user_message)
    for fact in expected.required_facts:
        missing = extract_numbers(fact, ignore_list_markers=False) - allowed
        if missing:
            warnings.append(f"fact '{fact}' has numbers not in the documentation: {sorted(missing)}")
    if expected.action == "answer" and not expected.required_facts:
        warnings.append("answer without required facts: only the judge can grade it")
    return warnings


def draft_one(item: EvalItem, provider, model: str, kb: dict) -> tuple[EvalItem, dict]:
    msgs = draft_messages(item, kb)
    stats = {"latency_s": 0.0, "attempts": 0, "cached": False}
    error = None
    for attempt in range(2):
        stats["attempts"] += 1
        res = provider.chat(model, msgs, json_schema=DRAFT_SCHEMA)
        stats["latency_s"] += res.latency_s
        stats["cached"] = res.cached
        try:
            data = parse_json_loose(res.text)
            if isinstance(data.get("tool"), dict) and not data["tool"].get("name"):
                data["tool"] = None
            expected = Expected.from_dict(data, where="draft")
            item.expected = expected
            item.draft_warnings = check_draft(expected, item, kb)
            return item, stats
        except (ValueError, SchemaError, AttributeError) as exc:
            error = str(exc)[:200]
            msgs = msgs + [
                {"role": "assistant", "content": res.text},
                {"role": "user", "content": f"That was not valid ({error}). Return only the JSON object."},
            ]
    item.expected = None
    item.draft_warnings = [f"draft failed: {error}"]
    return item, stats


def draft_all(items: list[EvalItem], provider, model: str, kb: dict, log=print) -> tuple[list[EvalItem], dict]:
    t0 = time.perf_counter()
    lat, failed, cached = [], 0, 0
    for i, item in enumerate(items, 1):
        item, st = draft_one(item, provider, model, kb)
        lat.append(st["latency_s"])
        cached += int(st["cached"])
        failed += int(item.expected is None)
        log(f"  draft {i}/{len(items)} {item.item_id} -> "
            f"{item.expected.action if item.expected else 'FAILED'} ({st['latency_s']:.1f}s{', cached' if st['cached'] else ''})")
    provider.release(model)
    return items, {
        "model": model,
        "items": len(items),
        "failed": failed,
        "with_warnings": sum(1 for it in items if it.draft_warnings),
        "cached": cached,
        "wall_s": round(time.perf_counter() - t0, 1),
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else 0.0,
    }
