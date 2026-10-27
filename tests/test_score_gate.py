import json
import tempfile
import unittest
from pathlib import Path

from evalgen.cli import main
from evalgen.gate import GateConfig, run_gate
from evalgen.io import write_jsonl
from evalgen.providers import ScriptedProvider
from evalgen.schema import EvalItem, Expected, ToolCall
from evalgen.score import deterministic_checks, judge_one, score_version, summarize
from evalgen.synth.kb import KB_DOCS


def item(expected, msg="How much is the Growth plan?", docs=("plans",), iid="i1"):
    return EvalItem(item_id=iid, log_id="l", user_message=msg, context_doc_ids=list(docs), cluster_id=0,
                    cluster_label="c", selection_reasons=["representative"], failure_signals=[], group_size=1,
                    expected=expected)


def pred(action="answer", message="", tool=None, iid="i1"):
    return {"item_id": iid, "version": "v", "response": {"action": action, "message": message, "tool": tool},
            "raw": "", "latency_s": 0.1}


GROWTH = Expected("answer", None, ["$29 per month", "up to 5 users"], "Growth costs $29 per month for up to 5 users.")


class TestDeterministic(unittest.TestCase):
    def test_correct(self):
        r = deterministic_checks(item(GROWTH), pred(message="Growth is $29 per month with up to 5 users."), KB_DOCS)
        self.assertEqual((r["det_score"], r["det_pass"]), (1.0, True))

    def test_hallucinated_number_is_penalised(self):
        r = deterministic_checks(item(GROWTH), pred(message="Growth is $29 per month, up to 5 users, $15 setup."), KB_DOCS)
        self.assertEqual(r["unsupported_numbers"], ["15"])
        self.assertEqual(r["det_score"], 0.5)

    def test_partial_and_refusal(self):
        r = deterministic_checks(item(GROWTH), pred(message="It is $29 per month."), KB_DOCS)
        self.assertEqual(r["fact_recall"], 0.5)
        r = deterministic_checks(item(GROWTH), pred(message="I'm sorry, but I can't help with that."), KB_DOCS)
        self.assertTrue(r["refusal"])
        self.assertEqual(r["det_score"], 0.0)

    def test_actions_and_tools(self):
        exp = Expected("tool_call", ToolCall("lookup_invoice", {"invoice_id": "INV-12345"}), [], "")
        it = item(exp, msg="is INV-12345 paid", docs=("invoice_lookup",))
        ok = pred("tool_call", "looking", {"name": "lookup_invoice", "arguments": {"invoice_id": "inv-12345"}})
        bad_arg = pred("tool_call", "looking", {"name": "lookup_invoice", "arguments": {"invoice_id": "INV-1"}})
        wrong = pred("tool_call", "human", {"name": "escalate_to_human", "arguments": {}})
        self.assertEqual(deterministic_checks(it, ok, KB_DOCS)["det_score"], 1.0)
        self.assertEqual(deterministic_checks(it, bad_arg, KB_DOCS)["det_score"], 0.0)
        self.assertEqual(deterministic_checks(it, wrong, KB_DOCS)["det_score"], 0.0)
        dec = item(Expected("decline", None, [], ""), msg="tell me a joke")
        self.assertEqual(deterministic_checks(dec, pred("decline", "Only Ledgerfox."), KB_DOCS)["det_score"], 1.0)
        self.assertEqual(deterministic_checks(dec, pred("answer", "Why did..."), KB_DOCS)["det_score"], 0.0)

    def test_invalid_json(self):
        p = {"item_id": "i1", "version": "v", "response": None, "raw": "oops", "latency_s": 0}
        r = deterministic_checks(item(GROWTH), p, KB_DOCS)
        self.assertFalse(r["valid_json"])
        judge = ScriptedProvider(lambda *a: self.fail("judge must not be called"))
        self.assertEqual(judge_one(item(GROWTH), p, judge, "j", KB_DOCS)[0], 1)


class TestJudgeAndSummary(unittest.TestCase):
    def test_judge_and_agreement(self):
        items = [item(GROWTH, iid="i1"), item(GROWTH, iid="i2")]
        preds = [pred(message="Growth is $29 per month for up to 5 users.", iid="i1"),
                 pred(message="It is $49 per month.", iid="i2")]
        judge = ScriptedProvider(lambda m, msgs, s: json.dumps(
            {"rationale": "r", "score": 2 if "$49" in msgs[-1]["content"] else 5}))
        scores = score_version(items, preds, KB_DOCS, judge=judge, judge_model="j", log=lambda *_: None)
        self.assertEqual([s.judge_score for s in scores], [5, 2])
        self.assertEqual(scores[0].combined, 1.0)
        summary = summarize(scores)
        self.assertEqual(summary["judge_det_agreement"], 1.0)
        self.assertEqual(summary["items"], 2)
        self.assertIn("0", summary["per_cluster"])

    def test_unusable_judge_output(self):
        judge = ScriptedProvider(lambda *a: '{"score": 9}')
        score, why = judge_one(item(GROWTH), pred(message="x"), judge, "j", KB_DOCS)
        self.assertIsNone(score)
        self.assertIn("unusable", why)


def rows(values, clusters=None, passes=None):
    out = []
    for i, v in enumerate(values):
        out.append({"item_id": f"i{i}", "cluster_id": (clusters or [0] * len(values))[i], "cluster_label": "c",
                    "combined": v, "det_score": v, "det_pass": (passes or [v >= 0.99] * len(values))[i],
                    "judge_score": 5 if v >= 0.99 else 2})
    return out


class TestGate(unittest.TestCase):
    def test_pass_on_identical(self):
        r = run_gate(rows([1, 1, 0.5, 1]), rows([1, 1, 0.5, 1]), GateConfig())
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["delta"], 0.0)

    def test_fail_on_overall_drop(self):
        r = run_gate(rows([1, 1, 1, 1]), rows([1, 1, 0.5, 0.5]), GateConfig(max_new_failures=10))
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("overall", r["reasons"][0])

    def test_cluster_gate_needs_enough_items(self):
        base = rows([1] * 10, clusters=[0] * 8 + [1] * 2)
        cand = rows([1] * 8 + [0.6, 0.6], clusters=[0] * 8 + [1] * 2)
        cfg = GateConfig(max_overall_drop=0.5, max_new_failures=10, min_cluster_items=3)
        self.assertEqual(run_gate(base, cand, cfg)["verdict"], "PASS")
        cfg.min_cluster_items = 2
        r = run_gate(base, cand, cfg)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any("cluster" in x for x in r["reasons"]))

    def test_new_failures(self):
        base = rows([1] * 20)
        cand = rows([1] * 16 + [0.99] * 4, passes=[True] * 16 + [False] * 4)
        r = run_gate(base, cand, GateConfig(max_new_failures=3))
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(len(r["new_failures"]), 4)

    def test_item_sets_must_match(self):
        with self.assertRaises(ValueError):
            run_gate(rows([1, 1]), rows([1]), GateConfig())

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "c.toml").write_text('[run]\nseed = 1\nrun_dir = "r"\n[gate]\nbaseline = "b"\n')
            write_jsonl(tmp / "r" / "04_scores_b.jsonl", rows([1, 1, 1, 1]))
            write_jsonl(tmp / "r" / "04_scores_good.jsonl", rows([1, 1, 1, 1]))
            write_jsonl(tmp / "r" / "04_scores_bad.jsonl", rows([0, 0, 1, 1]))
            args = ["--config", str(tmp / "c.toml"), "--run-dir", str(tmp / "r"), "gate"]
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--candidate", "good"]), 0)
                self.assertEqual(main(args + ["--candidate", "bad"]), 1)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(args + ["--candidate", "missing"]), 2)
            self.assertTrue((tmp / "r" / "05_gate_bad.json").exists())



class TestPromptVariants(unittest.TestCase):
    def test_no_docs_variant_hides_the_documentation(self):
        from evalgen.assistant import ModelVersion, build_messages

        msgs = build_messages(ModelVersion("x", "m", "v1-no-docs"), "price?", ["plans"], KB_DOCS)
        self.assertIn("(no documentation retrieved)", msgs[0]["content"])
        msgs = build_messages(ModelVersion("x", "m", "v1"), "price?", ["plans"], KB_DOCS)
        self.assertIn("$29 per month", msgs[0]["content"])
        with self.assertRaises(ValueError):
            ModelVersion("x", "m", "nope")


if __name__ == "__main__":
    unittest.main()
