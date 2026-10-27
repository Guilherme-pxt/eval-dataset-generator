"""Failure signals computed from a log record alone (no ground truth).

These are cheap heuristics a team could run over real traffic. They are
used to prioritise which cases enter the eval set, not to grade anything,
so false positives are acceptable: a reviewer sees every selected case.
"""

from __future__ import annotations

from .schema import LogRecord
from .synth.kb import context_text
from .textutil import extract_numbers, find_account_ids, find_invoice_ids, is_refusal

SIGNAL_WEIGHTS = {
    "thumbs_down": 1.0,
    "refusal": 2.0,
    "unsupported_number": 2.0,
    "possible_wrong_tool": 2.0,
    "possible_off_topic_compliance": 2.0,
}

DISPUTE_WORDS = ("chargeback", "dispute", "disputed")


def compute_signals(rec: LogRecord, kb: dict, off_topic: bool) -> list[str]:
    """Return the names of the signals that fire for ``rec``.

    ``off_topic`` is decided by the caller (it needs corpus statistics: the
    prompt's best similarity to any KB document).
    """
    signals = []
    resp = rec.response
    msg = resp.message or ""
    if rec.feedback == "down":
        signals.append("thumbs_down")
    if resp.action == "answer" and is_refusal(msg):
        signals.append("refusal")
    if resp.action == "answer":
        allowed = extract_numbers(context_text(rec.retrieved_doc_ids, kb)) | extract_numbers(rec.user_message)
        if extract_numbers(msg) - allowed:
            signals.append("unsupported_number")

    lower = rec.user_message.lower()
    tool = resp.tool.name if resp.tool else None
    if any(w in lower for w in DISPUTE_WORDS):
        if tool != "escalate_to_human":
            signals.append("possible_wrong_tool")
    elif find_invoice_ids(rec.user_message) and tool != "lookup_invoice":
        signals.append("possible_wrong_tool")
    elif find_account_ids(rec.user_message) and "refund" in lower and tool != "create_refund_request":
        signals.append("possible_wrong_tool")

    if off_topic and resp.action == "answer" and not is_refusal(msg):
        signals.append("possible_off_topic_compliance")
    return signals


def signal_score(signals: list[str]) -> float:
    return sum(SIGNAL_WEIGHTS.get(s, 0.0) for s in signals)
