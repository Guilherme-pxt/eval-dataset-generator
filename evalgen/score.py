"""Step 4: score model versions against the reviewed eval set.

Two independent graders per item:

* deterministic checks (cheap, exact, explainable)
    - action_match: answer / tool_call / decline as expected; for tools, the
      right tool and the right key argument (invoice_id, account_id)
    - fact_recall: share of required facts present in the reply
    - unsupported_numbers: numbers in the reply that appear neither in the
      documentation, the user message nor the reference answer
  det_score = 0 if the action is wrong, else fact_recall (1.0 for tool/decline),
  halved when the reply contains an unsupported number.

* LLM-as-judge with a fixed 1-5 rubric, normalised to 0-1.

combined = mean(det_score, judge_score). Both components are reported
separately, together with how often the two graders agree.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from .assistant import RESPONSE_SCHEMA, ModelVersion, build_messages, parse_response
from .review import KEY_ARGS
from .schema import AssistantResponse, EvalItem
from .synth.kb import context_text
from .textutil import extract_numbers, fact_present, is_refusal, parse_json_loose

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["rationale", "score"],
}

JUDGE_RUBRIC = """You grade one reply of the Ledgerfox customer support assistant.
The assistant must answer only from the documentation, decline requests unrelated to Ledgerfox,
and call a tool for invoice lookups (lookup_invoice), refunds on a specific account
(create_refund_request) and payment disputes (escalate_to_human).

You get the documentation, the user message, the reviewed expected behaviour and the reply.
Score the reply from 1 to 5:
5 = correct action (and correct tool), states every required fact, nothing unsupported or wrong.
4 = correct action, all key facts right, minor issue (verbosity, small omission of a secondary detail).
3 = correct action but a required fact is missing, or it adds a detail not in the documentation.
2 = correct action but a stated fact or number is wrong, or the reply is mostly unhelpful.
1 = wrong action: refuses an answerable question, answers an off-topic request, skips a required
    tool call or calls the wrong tool; or fabricates content.
Judge correctness against the documentation and the expected behaviour, not style.
Return JSON: {"rationale": "<one or two sentences>", "score": <1-5>}"""


@dataclass
class ItemScore:
    item_id: str
    version: str
    cluster_id: int
    cluster_label: str
    action: str
    expected_action: str
    valid_json: bool
    action_match: bool
    fact_recall: float
    unsupported_numbers: list[str]
    refusal: bool
    det_score: float
    det_pass: bool
    judge_score: int | None = None
    judge_norm: float | None = None
    judge_rationale: str = ""
    combined: float | None = None
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ predictions


def predict(items: list[EvalItem], version: ModelVersion, provider, kb: dict, log=print) -> list[dict]:
    preds = []
    t0 = time.perf_counter()
    for i, item in enumerate(items, 1):
        res = provider.chat(version.model, build_messages(version, item.user_message, item.context_doc_ids, kb),
                            json_schema=RESPONSE_SCHEMA)
        resp, err = parse_response(res.text)
        preds.append({
            "item_id": item.item_id,
            "version": version.name,
            "model": version.model,
            "prompt": version.prompt,
            "raw": res.text,
            "response": resp.to_dict() if resp else None,
            "parse_error": err,
            "latency_s": round(res.latency_s, 3),
            "cached": res.cached,
        })
        if i % 10 == 0 or i == len(items):
            log(f"  {version.name}: {i}/{len(items)} predictions ({time.perf_counter() - t0:.0f}s)")
    provider.release(version.model)
    return preds


# ------------------------------------------------------------------ deterministic


def deterministic_checks(item: EvalItem, pred: dict, kb: dict) -> dict:
    exp = item.expected
    resp = AssistantResponse.from_dict(pred["response"]) if pred.get("response") else None
    if resp is None:
        return {"valid_json": False, "action": "invalid", "action_match": False, "fact_recall": 0.0,
                "unsupported_numbers": [], "refusal": False, "det_score": 0.0, "det_pass": False}

    action_match = resp.action == exp.action
    tool_ok = True
    if action_match and exp.action == "tool_call":
        tool_ok = resp.tool is not None and resp.tool.name == exp.tool.name
        if tool_ok:
            for k in KEY_ARGS.get(exp.tool.name, ()):
                want = str(exp.tool.arguments.get(k, "")).strip().upper()
                got = str(resp.tool.arguments.get(k, "")).strip().upper()
                tool_ok = tool_ok and want == got
    action_match = action_match and tool_ok

    msg = resp.message or ""
    facts = exp.required_facts if exp.action == "answer" else []
    recall = sum(fact_present(f, msg) for f in facts) / len(facts) if facts else 1.0

    allowed = (extract_numbers(context_text(item.context_doc_ids, kb)) | extract_numbers(item.user_message)
               | extract_numbers(exp.reference_answer))
    unsupported = sorted(extract_numbers(msg) - allowed)
    refusal = resp.action == "answer" and is_refusal(msg)

    if not action_match:
        det = 0.0
    else:
        det = recall * (0.5 if unsupported else 1.0)
    return {
        "valid_json": True,
        "action": resp.action,
        "action_match": action_match,
        "fact_recall": round(recall, 4),
        "unsupported_numbers": unsupported,
        "refusal": refusal,
        "det_score": round(det, 4),
        "det_pass": det >= 0.999,
    }


# ------------------------------------------------------------------ judge


def judge_messages(item: EvalItem, pred: dict, kb: dict) -> list[dict]:
    exp = item.expected
    expected_txt = f"action: {exp.action}"
    if exp.tool:
        expected_txt += f"\ntool: {exp.tool.name} {exp.tool.arguments}"
    if exp.required_facts:
        expected_txt += "\nrequired facts: " + "; ".join(exp.required_facts)
    if exp.reference_answer:
        expected_txt += f"\nreference answer: {exp.reference_answer}"
    reply = pred.get("raw", "") if pred.get("response") is None else json.dumps(pred["response"], ensure_ascii=False)
    user = (
        f"Documentation:\n{context_text(item.context_doc_ids, kb)}\n\n"
        f"User message:\n{item.user_message}\n\n"
        f"Expected behaviour (reviewed by a human):\n{expected_txt}\n\n"
        f"Assistant reply to grade:\n{reply}"
    )
    return [{"role": "system", "content": JUDGE_RUBRIC}, {"role": "user", "content": user}]


def judge_one(item: EvalItem, pred: dict, provider, model: str, kb: dict) -> tuple[int | None, str]:
    if pred.get("response") is None:
        return 1, "reply is not valid JSON for the response schema"
    res = provider.chat(model, judge_messages(item, pred, kb), json_schema=JUDGE_SCHEMA)
    try:
        data = parse_json_loose(res.text)
        score = int(data["score"])
        if not 1 <= score <= 5:
            raise ValueError("score out of range")
        return score, str(data.get("rationale", ""))[:400]
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"judge output unusable: {exc}"


# ------------------------------------------------------------------ assemble


def score_version(items: list[EvalItem], preds: list[dict], kb: dict, judge=None, judge_model: str | None = None,
                  log=print) -> list[ItemScore]:
    """Deterministic checks for every item, plus the judge when ``judge`` (a provider) is given."""
    by_id = {p["item_id"]: p for p in preds}
    out = []
    for i, item in enumerate(items, 1):
        pred = by_id[item.item_id]
        det = deterministic_checks(item, pred, kb)
        s = ItemScore(
            item_id=item.item_id, version=pred["version"], cluster_id=item.cluster_id,
            cluster_label=item.cluster_label, expected_action=item.expected.action,
            latency_s=pred.get("latency_s", 0.0), **det,
        )
        if judge is not None:
            score, why = judge_one(item, pred, judge, judge_model, kb)
            s.judge_score, s.judge_rationale = score, why
            if score is not None:
                s.judge_norm = round((score - 1) / 4, 4)
                s.combined = round((s.det_score + s.judge_norm) / 2, 4)
            if i % 10 == 0 or i == len(items):
                log(f"  judged {pred['version']}: {i}/{len(items)}")
        out.append(s)
    return out


def summarize(scores: list[ItemScore]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    judged = [s for s in scores if s.judge_score is not None]
    agree = [s.det_pass == (s.judge_score >= 4) for s in judged]
    clusters = {}
    for s in scores:
        c = clusters.setdefault(s.cluster_id, {"label": s.cluster_label, "items": []})
        c["items"].append(s)
    per_cluster = {
        str(cid): {
            "label": c["label"],
            "n": len(c["items"]),
            "det_score": mean([s.det_score for s in c["items"]]),
            "judge_norm": mean([s.judge_norm for s in c["items"]]),
            "combined": mean([s.combined for s in c["items"]]),
        }
        for cid, c in sorted(clusters.items())
    }
    by_action = {}
    for a in ("answer", "tool_call", "decline"):
        sub = [s for s in scores if s.expected_action == a]
        if sub:
            by_action[a] = {"n": len(sub), "det_score": mean([s.det_score for s in sub]),
                            "combined": mean([s.combined for s in sub])}
    return {
        "items": len(scores),
        "valid_json_rate": mean([1.0 if s.valid_json else 0.0 for s in scores]),
        "action_accuracy": mean([1.0 if s.action_match else 0.0 for s in scores]),
        "det_score": mean([s.det_score for s in scores]),
        "det_pass_rate": mean([1.0 if s.det_pass else 0.0 for s in scores]),
        "items_with_unsupported_numbers": sum(1 for s in scores if s.unsupported_numbers),
        "refusals": sum(1 for s in scores if s.refusal),
        "judge_mean_1to5": mean([s.judge_score for s in judged]) if judged else None,
        "judge_norm": mean([s.judge_norm for s in scores]),
        "judge_failures": sum(1 for s in scores if s.judge_score is None) if judged else None,
        "combined": mean([s.combined for s in scores]),
        "judge_det_agreement": round(sum(agree) / len(agree), 4) if agree else None,
        "mean_latency_s": mean([s.latency_s for s in scores]),
        "by_expected_action": by_action,
        "per_cluster": per_cluster,
    }
