import json
import re
import tempfile
import unittest
from pathlib import Path

from evalgen.draft import draft_one
from evalgen.providers import ScriptedProvider
from evalgen.review import ReviewError, apply_decisions, load_decisions, oracle_decisions, render_review_html, review_cli
from evalgen.schema import EvalItem, Expected, ReviewDecision, SchemaError, ToolCall
from evalgen.synth.kb import KB_DOCS


def item(iid="item-001", log_id="log-1", msg="How much is the Growth plan?", docs=("plans",), expected=None):
    return EvalItem(item_id=iid, log_id=log_id, user_message=msg, context_doc_ids=list(docs), cluster_id=0,
                    cluster_label="growth / plan", selection_reasons=["representative"], failure_signals=[],
                    group_size=3, observed_response={"action": "answer", "message": "x", "tool": None},
                    expected=expected)


def answer(facts, ref="Growth costs $29 per month."):
    return Expected("answer", None, list(facts), ref)


class TestDraft(unittest.TestCase):
    def test_valid_draft_and_number_check(self):
        out = json.dumps({"action": "answer", "tool": None, "required_facts": ["$29 per month", "$99 setup fee"],
                          "reference_answer": "It is $29 per month."})
        it, stats = draft_one(item(), ScriptedProvider(lambda *a: out), "m", KB_DOCS)
        self.assertEqual(it.expected.required_facts, ["$29 per month", "$99 setup fee"])
        self.assertEqual(len(it.draft_warnings), 1)
        self.assertIn("99", it.draft_warnings[0])
        self.assertEqual(stats["attempts"], 1)

    def test_repair_then_give_up(self):
        replies = iter(["not json", json.dumps({"action": "decline", "tool": None, "required_facts": [],
                                                "reference_answer": "I only help with Ledgerfox."})])
        it, stats = draft_one(item(), ScriptedProvider(lambda *a: next(replies)), "m", KB_DOCS)
        self.assertEqual(it.expected.action, "decline")
        self.assertEqual(stats["attempts"], 2)
        it, _ = draft_one(item(), ScriptedProvider(lambda *a: '{"action": "fly"}'), "m", KB_DOCS)
        self.assertIsNone(it.expected)
        self.assertIn("draft failed", it.draft_warnings[0])


class TestApply(unittest.TestCase):
    def setUp(self):
        self.drafts = [item("a", expected=answer(["$29 per month"])), item("b", expected=answer(["x"])),
                       item("c", expected=None)]

    def test_accept_edit_reject(self):
        decisions = [ReviewDecision("a", "accept"), ReviewDecision("b", "reject"),
                     ReviewDecision("c", "edit", Expected("decline", None, [], "no"), "fixed", "me")]
        ev, stats = apply_decisions(self.drafts, decisions)
        self.assertEqual([i.item_id for i in ev], ["a", "c"])
        self.assertEqual(ev[1].expected.action, "decline")
        self.assertEqual(ev[1].review, {"decision": "edit", "note": "fixed", "reviewer": "me"})
        self.assertEqual(stats, {"accept": 1, "edit": 1, "reject": 1, "undecided": 0})

    def test_errors(self):
        with self.assertRaises(ReviewError):
            apply_decisions(self.drafts, [ReviewDecision("zzz", "accept")])
        with self.assertRaises(ReviewError):
            apply_decisions(self.drafts, [ReviewDecision("a", "accept")])
        ev, stats = apply_decisions(self.drafts, [ReviewDecision("a", "accept")], allow_partial=True)
        self.assertEqual(stats["undecided"], 2)
        with self.assertRaises(ReviewError):
            apply_decisions(self.drafts, [ReviewDecision("c", "accept"), ReviewDecision("a", "accept"),
                                          ReviewDecision("b", "accept")])
        with self.assertRaises(SchemaError):
            ReviewDecision.from_dict({"item_id": "a", "decision": "edit"})


class TestOracle(unittest.TestCase):
    def truth(self, **kw):
        base = {"ambiguous": False, "expected_action": "answer", "expected_tool": None,
                "gold_facts": ["$29 per month"], "gold_reference": "Growth costs $29 per month."}
        base.update(kw)
        return base

    def test_decisions(self):
        drafts = [
            item("ok", "l1", expected=answer(["$29 per month"])),
            item("extra", "l2", expected=answer(["$29 per month", "up to 25 users"])),
            item("wrong", "l3", expected=answer(["$29 per month"])),
            item("amb", "l4", expected=answer([])),
            item("tool", "l5", expected=Expected("tool_call", ToolCall("lookup_invoice", {"invoice_id": "INV-1"}), [], "")),
        ]
        truth = {
            "l1": self.truth(), "l2": self.truth(), "l3": self.truth(expected_action="decline", gold_facts=[]),
            "l4": self.truth(ambiguous=True, expected_action=None),
            "l5": self.truth(expected_action="tool_call", gold_facts=[],
                             expected_tool={"name": "lookup_invoice", "arguments": {"invoice_id": "INV-2"}}),
        }
        d = {x.item_id: x for x in oracle_decisions(drafts, truth)}
        self.assertEqual(d["ok"].decision, "accept")
        self.assertEqual(d["extra"].decision, "edit")
        self.assertIn("not needed", d["extra"].note)
        self.assertEqual(d["extra"].expected.required_facts, ["$29 per month"])
        self.assertEqual(d["wrong"].decision, "edit")
        self.assertEqual(d["wrong"].expected.action, "decline")
        self.assertEqual(d["amb"].decision, "reject")
        self.assertEqual(d["tool"].decision, "edit")
        self.assertEqual(d["tool"].expected.tool.arguments, {"invoice_id": "INV-2"})


class TestCliAndHtml(unittest.TestCase):
    def test_cli_review_is_resumable(self):
        drafts = [item("a", expected=answer(["$29 per month"])), item("b", expected=answer(["x"])),
                  item("c", expected=answer(["y"]))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            # a: accept; b: edit to decline; then quit before c
            answers = iter(["a", "", "e", "decline", "", "nope", "q"])
            review_cli(drafts, path, KB_DOCS, ask=lambda _: next(answers), out=lambda _: None)
            decisions = load_decisions(path)
            self.assertEqual([(x.item_id, x.decision) for x in decisions], [("a", "accept"), ("b", "edit")])
            self.assertEqual(decisions[1].expected.action, "decline")
            self.assertEqual(decisions[1].expected.reference_answer, drafts[1].expected.reference_answer)
            answers = iter(["r", "dup"])
            review_cli(drafts, path, KB_DOCS, ask=lambda _: next(answers), out=lambda _: None)
            self.assertEqual([x.item_id for x in load_decisions(path)], ["a", "b", "c"])

    def test_html_embeds_safe_json(self):
        it = item(expected=answer(["$29 per month"], ref="contains </script> on purpose"))
        html = render_review_html([it], KB_DOCS)
        m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.DOTALL)
        data = json.loads(m.group(1))
        self.assertEqual(data["items"][0]["expected"]["reference_answer"], "contains </script> on purpose")
        self.assertIn("Plans and pricing", data["items"][0]["context"])
        self.assertNotIn("__REVIEW_DATA__", html)


if __name__ == "__main__":
    unittest.main()
