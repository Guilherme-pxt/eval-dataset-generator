"""Step 3b: human review of the drafts.

Three ways to produce a decisions file (a JSON list of ``ReviewDecision``):

* ``review cli``    interactive terminal loop, resumable, saves after each decision
* ``review html``   writes a self-contained static page; the reviewer exports decisions.json
* ``review oracle`` scripted reviewer that knows the synthetic ground truth; it makes
                    reference runs reproducible and measures how good the drafts were

``review apply`` validates a decisions file against the drafts and writes the
reviewed eval set (accepted + edited items only).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .io import read_json, write_json
from .schema import TOOLS, EvalItem, Expected, ReviewDecision, SchemaError, ToolCall
from .synth.kb import context_text
from .textutil import fact_present

KEY_ARGS = {"lookup_invoice": ("invoice_id",), "create_refund_request": ("account_id",), "escalate_to_human": ()}
MAX_FACT_WORDS = 8


class ReviewError(ValueError):
    pass


# ------------------------------------------------------------------ apply


def load_decisions(path: str | Path) -> list[ReviewDecision]:
    raw = read_json(path)
    if isinstance(raw, dict) and "decisions" in raw:
        raw = raw["decisions"]
    if not isinstance(raw, list):
        raise ReviewError("decisions file must be a JSON list (or an object with a 'decisions' list)")
    return [ReviewDecision.from_dict(d) for d in raw]


def apply_decisions(drafts: list[EvalItem], decisions: list[ReviewDecision], allow_partial: bool = False):
    by_id = {d.item_id: d for d in drafts}
    seen = {}
    for dec in decisions:
        if dec.item_id not in by_id:
            raise ReviewError(f"decision for unknown item '{dec.item_id}'")
        seen[dec.item_id] = dec  # last decision wins (the CLI appends on re-review)
    missing = [i for i in by_id if i not in seen]
    if missing and not allow_partial:
        raise ReviewError(f"{len(missing)} item(s) have no decision, e.g. {missing[:3]} (use --allow-partial)")

    eval_set, stats = [], {"accept": 0, "edit": 0, "reject": 0, "undecided": len(missing)}
    for item in drafts:
        dec = seen.get(item.item_id)
        if dec is None:
            continue
        stats[dec.decision] += 1
        if dec.decision == "reject":
            continue
        if dec.decision == "accept":
            if item.expected is None:
                raise ReviewError(f"{item.item_id}: cannot accept an item whose draft failed; edit it instead")
            expected = item.expected
        else:
            expected = dec.expected
        reviewed = EvalItem.from_dict(item.to_dict())
        reviewed.expected = expected
        reviewed.review = {"decision": dec.decision, "note": dec.note, "reviewer": dec.reviewer}
        eval_set.append(reviewed)
    return eval_set, stats


# ------------------------------------------------------------------ oracle


def oracle_decisions(drafts: list[EvalItem], truth: dict[str, dict]) -> list[ReviewDecision]:
    """Simulated reviewer with access to the synthetic ground truth.

    accept: the draft has the right action/tool, lists every gold fact in short,
            checkable form, and asks for nothing the question did not ask
    edit:   anything else that is fixable; the gold fields replace the draft's
    reject: the request is ambiguous, so no single expected output exists
    """
    out = []
    for item in drafts:
        gt = truth[item.log_id]
        if gt["ambiguous"]:
            out.append(ReviewDecision(item.item_id, "reject", None, "ambiguous request: no single correct output", "oracle"))
            continue
        problems = []
        exp = item.expected
        if exp is None:
            problems.append("draft failed")
        else:
            if exp.action != gt["expected_action"]:
                problems.append(f"action {exp.action} should be {gt['expected_action']}")
            elif exp.action == "tool_call":
                gold_tool = gt["expected_tool"]
                if exp.tool.name != gold_tool["name"]:
                    problems.append(f"tool {exp.tool.name} should be {gold_tool['name']}")
                for k in KEY_ARGS[gold_tool["name"]]:
                    if str(exp.tool.arguments.get(k, "")).upper() != str(gold_tool["arguments"][k]).upper():
                        problems.append(f"argument {k} is wrong")
            elif exp.action == "answer":
                facts_blob = " ; ".join(exp.required_facts)
                for f in gt["gold_facts"]:
                    if not fact_present(f, facts_blob):
                        problems.append(f"missing required fact '{f}'")
                for f in exp.required_facts:
                    if len(f.split()) > MAX_FACT_WORDS:
                        problems.append(f"fact too long to check reliably: '{f[:40]}...'")
                    elif not any(fact_present(g, f) or fact_present(f, g) for g in gt["gold_facts"]):
                        problems.append(f"fact not needed for this question: '{f}'")
            problems += [w for w in item.draft_warnings if "numbers not in the documentation" in w]
        if not problems:
            out.append(ReviewDecision(item.item_id, "accept", None, "", "oracle"))
            continue
        gold_tool = gt["expected_tool"]
        tool = None
        if gold_tool:
            tool = ToolCall(gold_tool["name"], {k: gold_tool["arguments"][k] for k in KEY_ARGS[gold_tool["name"]]})
        keep_reference = exp is not None and exp.action == gt["expected_action"] and exp.reference_answer
        edited = Expected(
            action=gt["expected_action"],
            tool=tool,
            required_facts=list(gt["gold_facts"]) if gt["expected_action"] == "answer" else [],
            reference_answer=exp.reference_answer if keep_reference else gt["gold_reference"],
        )
        out.append(ReviewDecision(item.item_id, "edit", edited, "; ".join(problems), "oracle"))
    return out


# ------------------------------------------------------------------ CLI


def _show(item: EvalItem, kb: dict, out: Callable[[str], None]) -> None:
    out("=" * 72)
    out(f"{item.item_id}  cluster: {item.cluster_label}  reasons: {', '.join(item.selection_reasons)}")
    if item.failure_signals:
        out(f"signals: {', '.join(item.failure_signals)}")
    out(f"\nUSER: {item.user_message}")
    out(f"\nDOCS:\n{context_text(item.context_doc_ids, kb)}")
    if item.observed_response:
        out(f"\nPRODUCTION REPLY (may be wrong): {json.dumps(item.observed_response, ensure_ascii=False)}")
    if item.expected:
        e = item.expected
        out(f"\nDRAFT  action: {e.action}  tool: {json.dumps(e.tool.__dict__) if e.tool else None}")
        out(f"       facts: {e.required_facts}")
        out(f"       reference: {e.reference_answer}")
    for w in item.draft_warnings:
        out(f"  ! {w}")


def _edit(item: EvalItem, ask: Callable[[str], str]) -> Expected:
    cur = item.expected or Expected("answer", None, [], "")

    def field(label, default):
        v = ask(f"{label} [{default}]: ").strip()
        return v if v else default

    while True:
        action = field("action (answer/tool_call/decline)", cur.action)
        tool = None
        if action == "tool_call":
            name = field(f"tool ({'/'.join(TOOLS)})", cur.tool.name if cur.tool else "lookup_invoice")
            args = {}
            for k in KEY_ARGS.get(name, ()):
                args[k] = field(k, (cur.tool.arguments.get(k, "") if cur.tool else ""))
            tool = {"name": name, "arguments": args}
        facts_default = " | ".join(cur.required_facts)
        facts = field("required facts, separated by ' | '", facts_default) if action == "answer" else ""
        reference = field("reference answer", cur.reference_answer)
        try:
            return Expected.from_dict({
                "action": action, "tool": tool,
                "required_facts": [f.strip() for f in facts.split("|") if f.strip()],
                "reference_answer": reference,
            })
        except SchemaError as exc:
            ask(f"invalid: {exc}. Press enter to retry.")


def review_cli(drafts: list[EvalItem], decisions_path: str | Path, kb: dict,
               ask: Callable[[str], str] = input, out: Callable[[str], None] = print) -> list[ReviewDecision]:
    path = Path(decisions_path)
    decisions = load_decisions(path) if path.exists() else []
    done = {d.item_id for d in decisions}
    todo = [it for it in drafts if it.item_id not in done]
    out(f"{len(done)} already reviewed, {len(todo)} to go. Decisions are saved to {path} after each item.")
    for item in todo:
        _show(item, kb, out)
        while True:
            choice = ask("\n[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()
            if choice in ("a", "e", "r", "s", "q"):
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "a" and item.expected is None:
            out("draft failed; use [e]dit to write the expected output")
            choice = "e"
        expected = _edit(item, ask) if choice == "e" else None
        note = ask("note (optional): ").strip()
        decision = {"a": "accept", "e": "edit", "r": "reject"}[choice]
        decisions.append(ReviewDecision(item.item_id, decision, expected, note, "human:cli"))
        write_json(path, [d.to_dict() for d in decisions])
    return decisions


# ------------------------------------------------------------------ static HTML


def render_review_html(drafts: list[EvalItem], kb: dict, title: str = "Eval set review") -> str:
    template = (Path(__file__).parent / "review_page.html").read_text(encoding="utf-8")
    payload = []
    for it in drafts:
        d = it.to_dict()
        d["context"] = context_text(it.context_doc_ids, kb)
        payload.append(d)
    data = json.dumps({
        "title": title,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tools": list(TOOLS),
        "key_args": {k: list(v) for k, v in KEY_ARGS.items()},
        "items": payload,
    }, ensure_ascii=False)
    data = data.replace("</", "<\\/")  # never close the <script> tag early
    return template.replace("__REVIEW_DATA__", data)
