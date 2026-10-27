import json
import tempfile
import unittest
from pathlib import Path

from evalgen.ingest import ingest
from evalgen.schema import LogRecord
from evalgen.signals import compute_signals
from evalgen.synth.kb import KB_DOCS


def rec(i, msg, action="answer", message="ok", tool=None, ts="2026-09-02T10:00:00Z", version="assistant-v3.1",
        docs=("plans",), feedback=None):
    return {"id": f"log-{i}", "timestamp": ts, "model_version": version, "user_message": msg,
            "retrieved_doc_ids": list(docs), "response": {"action": action, "message": message, "tool": tool},
            "feedback": feedback}


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "log.jsonl"
        lines = [
            json.dumps(rec(1, "how much is growth? mail me at a.b@example.com")),
            '{"id": "log-2", "broken',
            json.dumps(rec(3, "is INV-12345 paid", ts="2026-09-10T10:00:00Z", version="assistant-v3.2",
                           docs=("invoice_lookup", "nope"))),
            json.dumps({**rec(4, "x"), "response": {"action": "dance", "message": ""}}),
            json.dumps(rec(1, "duplicate id")),
            json.dumps(rec(5, "   ")),
        ]
        self.path.write_text("\n".join(lines) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation_redaction_and_doc_ids(self):
        out = Path(self.tmp.name) / "clean.jsonl"
        records, report = ingest(self.path, KB_DOCS, out_path=out)
        self.assertEqual([r.id for r in records], ["log-1", "log-3"])
        self.assertEqual(report.read, 6)
        self.assertEqual(len(report.rejected), 4)
        reasons = " ".join(r["reason"] for r in report.rejected)
        for needle in ("invalid JSON", "action must be one of", "duplicate id", "user_message is empty"):
            self.assertIn(needle, reasons)
        self.assertIn("[EMAIL]", records[0].user_message)
        self.assertEqual(report.pii_redactions, 1)
        self.assertEqual(records[1].retrieved_doc_ids, ["invoice_lookup"])
        self.assertEqual(report.unknown_doc_ids, 1)
        self.assertEqual(len(out.read_text().splitlines()), 2)

    def test_filters(self):
        recs, _ = ingest(self.path, KB_DOCS, since="2026-09-05T00:00:00Z")
        self.assertEqual([r.id for r in recs], ["log-3"])
        recs, _ = ingest(self.path, KB_DOCS, model_version="assistant-v3.1")
        self.assertEqual([r.id for r in recs], ["log-1"])
        recs, _ = ingest(self.path, KB_DOCS, limit=1)
        self.assertEqual(len(recs), 1)


class TestSignals(unittest.TestCase):
    def sig(self, d, off_topic=False):
        return compute_signals(LogRecord.from_dict(d), KB_DOCS, off_topic)

    def test_clean_answer_has_no_signal(self):
        self.assertEqual(self.sig(rec(1, "growth price?", message="Growth costs $29 per month.")), [])

    def test_unsupported_number_and_thumbs_down(self):
        s = self.sig(rec(1, "growth price?", message="Growth costs $39 per month.", feedback="down"))
        self.assertEqual(s, ["thumbs_down", "unsupported_number"])

    def test_refusal(self):
        self.assertIn("refusal", self.sig(rec(1, "growth price?", message="I'm sorry, but I can't help with that.")))

    def test_wrong_tool(self):
        self.assertIn("possible_wrong_tool", self.sig(rec(1, "is INV-12345 paid", message="Check the Invoices tab.")))
        ok = rec(1, "is INV-12345 paid", action="tool_call", message="looking",
                 tool={"name": "lookup_invoice", "arguments": {"invoice_id": "INV-12345"}})
        self.assertNotIn("possible_wrong_tool", self.sig(ok))
        dispute = rec(1, "chargeback on INV-12345", action="tool_call", message="looking",
                      tool={"name": "lookup_invoice", "arguments": {"invoice_id": "INV-12345"}})
        self.assertIn("possible_wrong_tool", self.sig(dispute))

    def test_off_topic_compliance(self):
        self.assertIn("possible_off_topic_compliance", self.sig(rec(1, "tell me a joke", message="Why did..."), True))
        declined = rec(1, "tell me a joke", action="decline", message="I can only help with Ledgerfox.")
        self.assertEqual(self.sig(declined, True), [])


if __name__ == "__main__":
    unittest.main()
