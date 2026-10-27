"""Shared fixtures: a fake LLM that behaves like a drafter, an assistant and a judge."""

from __future__ import annotations

import json
import re

from evalgen.providers import ScriptedProvider
from evalgen.textutil import redact_pii


def item_for(messages: list[dict]) -> str:
    """The user message of an assistant prompt, or the one embedded in a draft/judge prompt."""
    text = messages[-1]["content"]
    m = re.search(r"User message:\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def make_fake(truth_by_message: dict[str, dict], bad_models: tuple[str, ...] = ("bad-model",)) -> ScriptedProvider:
    """A deterministic stand-in for Ollama.

    * drafter: right action and tool, but always adds an unrequested fact, so the
      oracle has something to edit
    * assistant: replies with the gold answer, unless the model is in ``bad_models``
      (then it refuses everything)
    * judge: 5 for anything that is not a refusal, 1 otherwise
    """

    def fn(model, messages, schema):
        system = messages[0]["content"]
        if system.startswith("You write the expected output"):
            gt = truth_by_message[item_for(messages)]
            action = gt["expected_action"] or "answer"
            return json.dumps({
                "action": action,
                "tool": gt["expected_tool"],
                "required_facts": list(gt["gold_facts"]) + (["Ledgerfox"] if action == "answer" else []),
                "reference_answer": gt["gold_reference"],
            })
        if system.startswith("You grade one reply"):
            reply = messages[-1]["content"].split("Assistant reply to grade:\n", 1)[-1]
            score = 1 if "can't help" in reply else 5
            return json.dumps({"rationale": "fake judge", "score": score})
        gt = truth_by_message[item_for(messages)]
        if model in bad_models:
            return json.dumps({"action": "answer", "tool": None, "message": "I'm sorry, but I can't help with that."})
        return json.dumps({
            "action": gt["expected_action"] or "answer",
            "tool": gt["expected_tool"],
            "message": gt["gold_reference"],
        })

    return ScriptedProvider(fn)


def truth_by_message(lines: list[str], truths: list[dict]) -> dict[str, dict]:
    by_id = {t["id"]: t for t in truths}
    out = {}
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("id") in by_id and "user_message" in rec:
            out[redact_pii(rec["user_message"])[0].strip()] = by_id[rec["id"]]
    return out
