import json
import unittest
from collections import Counter

from evalgen.schema import LogRecord
from evalgen.synth.generate import generate
from evalgen.synth.intents import INTENTS
from evalgen.synth.kb import KB_DOCS, context_text
from evalgen.textutil import extract_numbers, fact_present


class TestSynth(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        self.assertEqual(generate(120, seed=7), generate(120, seed=7))
        self.assertNotEqual(generate(120, seed=7)[0], generate(120, seed=8)[0])

    def test_shape_and_failures(self):
        lines, truths = generate(300, seed=42)
        self.assertEqual(len(truths), 300)
        self.assertEqual(len(lines), 303)  # three malformed lines on purpose
        kinds = Counter(t["failure_type"] for t in truths)
        for k in ("hallucinated_number", "refusal", "wrong_tool", "off_topic_compliance"):
            self.assertGreater(kinds[k], 0, k)
        valid = 0
        for line in lines:
            try:
                LogRecord.from_dict(json.loads(line))
                valid += 1
            except Exception:
                pass
        self.assertEqual(valid, 300)

    def test_gold_is_consistent_with_the_documentation(self):
        """Every gold fact is stated in the gold reply and its numbers exist in the KB."""
        _, truths = generate(300, seed=42)
        intents = {i.id: i for i in INTENTS}
        for t in truths:
            for fact in t["gold_facts"]:
                self.assertTrue(fact_present(fact, t["gold_reference"]), (t["intent"], fact))
                docs = context_text(list(intents[t["intent"]].docs), KB_DOCS)
                self.assertLessEqual(extract_numbers(fact, ignore_list_markers=False), extract_numbers(docs), fact)

    def test_hallucinations_introduce_unsupported_numbers(self):
        lines, truths = generate(300, seed=42)
        by_id = {json.loads(l)["id"]: json.loads(l) for l in lines if l.startswith('{"id": "log-0')}
        for t in truths:
            if t["failure_type"] == "hallucinated_number":
                rec = by_id[t["id"]]
                allowed = extract_numbers(context_text(rec["retrieved_doc_ids"], KB_DOCS))
                self.assertTrue(extract_numbers(rec["response"]["message"]) - allowed, rec["response"]["message"])


if __name__ == "__main__":
    unittest.main()
