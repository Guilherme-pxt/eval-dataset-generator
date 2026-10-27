"""Seeded generator for a synthetic production log of the Ledgerfox assistant.

Outputs:
  production_log.jsonl  what a logging pipeline would capture (the pipeline input)
  ground_truth.jsonl    hidden labels per record (intent, gold answer, injected failure)
  kb.json               the documentation the assistant was grounded on

The same seed always yields byte-identical files. No LLM is involved, so the
dataset is reproducible offline.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..io import write_json, write_jsonl
from .intents import DEFAULT_ANSWER_FAILURES, DEFAULT_TOOL_FAILURES, INTENTS, Intent
from ..textutil import extract_numbers, normalize_number
from .kb import KB_DOCS, context_text

PREFIXES = [("", 0.55), ("hi, ", 0.1), ("hey ", 0.08), ("hello! ", 0.07), ("quick question: ", 0.1), ("hi team, ", 0.1)]
SUFFIXES = [("", 0.6), (" thanks", 0.15), (" thx", 0.08), (" please", 0.1), ("??", 0.07)]
CITIES = ["Lisbon", "Toronto", "Berlin", "Austin", "Melbourne", "Dublin"]
FIRST_NAMES = ["ana", "ben", "carla", "dev", "elena", "farid", "grace", "hugo"]
REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I'm unable to provide that information.",
]
WRONG_TOOL_ANSWERS = [
    "You can see the status of every invoice in the Invoices tab.",
    "Please contact our support team by email and they will look into it.",
]
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
DAYS = 14


def _weighted(rng: random.Random, pairs):
    values, weights = zip(*pairs)
    return rng.choices(values, weights=weights, k=1)[0]


def _typo(rng: random.Random, text: str) -> str:
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def _wrong_number(rng: random.Random, token: str, avoid: set[str]) -> str:
    """Plausible but wrong replacement for a number token, never one the docs contain."""
    if "." in token:
        whole, frac = token.split(".")
        options = [f"{int(whole) + d}.{frac}" for d in (1, 2, 3)]
    else:
        n = int(token)
        options = [str(o) for o in ([n * 2, n + 10, n + 20] if n >= 10 else [n + 2, n + 3, n * 3, n + 10])]
    options = [o for o in options if o != token and normalize_number(o) not in avoid]
    return rng.choice(options)


def _hallucinate(rng: random.Random, message: str, avoid: set[str]) -> str:
    nums = list(re.finditer(r"\d+(?:\.\d+)?", message))
    m = rng.choice(nums)
    return message[: m.start()] + _wrong_number(rng, m.group(0), avoid) + message[m.end() :]


def _failure_rates(intent: Intent) -> dict:
    if intent.failure_rates:
        rates = dict(intent.failure_rates)
    elif intent.action == "answer":
        rates = dict(DEFAULT_ANSWER_FAILURES)
    elif intent.action == "tool_call":
        rates = dict(DEFAULT_TOOL_FAILURES)
    else:
        rates = {}
    if not any(re.search(r"\d", a) for a in intent.answers):
        rates.pop("hallucinated_number", None)  # nothing to get wrong
    return rates


def _make_record(rng: random.Random, intent: Intent) -> tuple[dict, dict]:
    t_idx = rng.randrange(len(intent.templates))
    slots = {
        "inv": f"INV-{rng.randint(10000, 99999)}",
        "acc": f"ACC-{rng.randint(10000, 99999)}",
        "city": rng.choice(CITIES),
    }
    core = intent.templates[t_idx].format(**slots)
    text = _weighted(rng, PREFIXES) + core + _weighted(rng, SUFFIXES)
    if rng.random() < 0.3:
        text = text.lower()
    if rng.random() < 0.1:
        text = _typo(rng, text)
    if rng.random() < 0.05:
        text += f" (you can reach me at {rng.choice(FIRST_NAMES)}.{rng.randint(1, 99)}@example.com)"

    docs = list(intent.docs)
    if not docs:  # the retriever always returns something, even for off-topic turns
        docs = [rng.choice(sorted(KB_DOCS))]
    elif rng.random() < 0.25:
        extra = rng.choice([d for d in sorted(KB_DOCS) if d not in docs])
        docs.append(extra)

    gold_message = rng.choice(intent.answers).format(**slots) if intent.answers else ""
    gold_tool = None
    if intent.action == "tool_call":
        args = {}
        if intent.tool == "lookup_invoice":
            args = {"invoice_id": slots["inv"]}
        elif intent.tool == "create_refund_request":
            args = {"account_id": slots["acc"], "reason": core}
        elif intent.tool == "escalate_to_human":
            args = {"reason": "payment dispute"}
        gold_tool = {"name": intent.tool, "arguments": args}

    failure = None
    roll = rng.random()
    acc = 0.0
    for mode, p in sorted(_failure_rates(intent).items()):
        acc += p
        if roll < acc:
            failure = mode
            break

    gold_action = intent.action if intent.action != "ambiguous" else "answer"
    response = {"action": gold_action, "message": gold_message, "tool": gold_tool}
    if failure == "hallucinated_number":
        avoid = extract_numbers(context_text(docs, KB_DOCS)) | extract_numbers(text)
        response["message"] = _hallucinate(rng, gold_message, avoid)
    elif failure == "refusal":
        response = {"action": "answer", "message": rng.choice(REFUSALS), "tool": None}
    elif failure == "wrong_tool":
        if rng.random() < 0.5 and intent.tool != "escalate_to_human":
            response = {
                "action": "tool_call",
                "message": "Let me get a human to help you with this.",
                "tool": {"name": "escalate_to_human", "arguments": {"reason": core}},
            }
        elif intent.tool == "escalate_to_human" and "INV-" in core:
            response = {
                "action": "tool_call",
                "message": f"Let me look up invoice {slots['inv']} for you.",
                "tool": {"name": "lookup_invoice", "arguments": {"invoice_id": slots["inv"]}},
            }
        else:
            response = {"action": "answer", "message": rng.choice(WRONG_TOOL_ANSWERS), "tool": None}
    elif failure == "off_topic_compliance":
        response = {"action": "answer", "message": intent.compliance[t_idx].format(**slots), "tool": None}

    if failure:
        feedback = _weighted(rng, [("down", 0.5), (None, 0.45), ("up", 0.05)])
    else:
        feedback = _weighted(rng, [(None, 0.7), ("up", 0.25), ("down", 0.05)])

    record = {
        "timestamp": None,
        "session_id": f"s-{rng.randint(100000, 999999)}",
        "model_version": None,
        "user_message": text,
        "retrieved_doc_ids": docs,
        "response": response,
        "feedback": feedback,
        "latency_ms": int(rng.lognormvariate(7.0, 0.35)),
    }
    truth = {
        "intent": intent.id,
        "ambiguous": intent.action == "ambiguous",
        "expected_action": None if intent.action == "ambiguous" else intent.action,
        "expected_tool": gold_tool,
        "gold_facts": list(intent.template_facts.get(t_idx, intent.gold_facts)),
        "gold_reference": gold_message,
        "failure_type": failure,
    }
    return record, truth


def generate(n_records: int = 300, seed: int = 42, n_malformed: int = 3) -> tuple[list[str], list[dict]]:
    """Return (raw JSONL lines of the production log, ground-truth rows)."""
    rng = random.Random(seed)
    weights = [i.weight for i in INTENTS]
    pairs = []
    for _ in range(n_records):
        intent = rng.choices(INTENTS, weights=weights, k=1)[0]
        offset = timedelta(seconds=rng.randrange(DAYS * 24 * 3600))
        pairs.append((offset, intent))
    pairs.sort(key=lambda p: p[0])

    lines, truths = [], []
    for idx, (offset, intent) in enumerate(pairs):
        record, truth = _make_record(rng, intent)
        rid = f"log-{idx + 1:05d}"
        ts = START + offset
        record = {
            "id": rid,
            **record,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": "assistant-v3.1" if offset < timedelta(days=7) else "assistant-v3.2",
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        truths.append({"id": rid, **truth})

    # Real logs are never clean: inject a few malformed lines that ingest must reject.
    bad = [
        '{"id": "log-bad-1", "timestamp": "2026-09-03T10:00:00Z", "user_message": "how much is',
        json.dumps({"id": "log-bad-2", "timestamp": "2026-09-05T12:00:00Z", "model_version": "assistant-v3.1",
                    "retrieved_doc_ids": [], "response": {"action": "answer", "message": "ok"}}),
        json.dumps({"id": "log-bad-3", "timestamp": "2026-09-09T08:30:00Z", "model_version": "assistant-v3.2",
                    "user_message": "cancel my plan", "retrieved_doc_ids": ["plans"],
                    "response": {"action": "shrug", "message": ""}}),
    ][:n_malformed]
    for b in bad:
        lines.insert(rng.randrange(len(lines) + 1), b)
    return lines, truths


def write_dataset(out_dir: str | Path, n_records: int = 300, seed: int = 42) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines, truths = generate(n_records=n_records, seed=seed)
    (out / "production_log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_jsonl(out / "ground_truth.jsonl", truths)
    write_json(out / "kb.json", KB_DOCS)
    failures = {}
    for t in truths:
        key = t["failure_type"] or "none"
        failures[key] = failures.get(key, 0) + 1
    intents = {}
    for t in truths:
        intents[t["intent"]] = intents.get(t["intent"], 0) + 1
    return {
        "records": len(truths),
        "malformed_lines": len(lines) - len(truths),
        "intents": len(intents),
        "failure_counts": dict(sorted(failures.items())),
        "seed": seed,
    }
