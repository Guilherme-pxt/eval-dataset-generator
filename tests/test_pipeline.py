"""End-to-end run with a fake provider: no network, no Ollama."""

import tempfile
import unittest
from pathlib import Path

from evalgen import pipeline as P
from evalgen.io import read_json, read_jsonl
from evalgen.synth.generate import generate

from .helpers import make_fake, truth_by_message


def config(tmp: Path) -> dict:
    return {
        "run": {"seed": 42, "data_dir": str(tmp / "data"), "run_dir": str(tmp / "run")},
        "synth": {"records": 150},
        "ingest": {"log": str(tmp / "data" / "production_log.jsonl")},
        "cluster": {"budget": 16, "k_min": 8, "k_max": 14},
        "llm": {"provider": "ollama"},
        "draft": {"model": "drafter"},
        "review": {"mode": "oracle"},
        "judge": {"model": "judge"},
        "versions": {
            "baseline": {"model": "good-model", "prompt": "v1"},
            "same": {"model": "good-model-2", "prompt": "v1"},
            "broken": {"model": "bad-model", "prompt": "v1-no-policy"},
        },
        "gate": {"baseline": "baseline", "max_overall_drop": 0.05, "max_cluster_drop": 0.15,
                 "min_cluster_items": 3, "max_new_failures": 3},
    }


class TestRunAll(unittest.TestCase):
    def test_end_to_end_with_fake_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = config(tmp)
            lines, truths = generate(150, seed=42)
            fake = make_fake(truth_by_message(lines, truths))
            results = P.run_all(cfg, tmp / "run", fake, log=lambda *_: None)

            self.assertEqual(results["same"]["verdict"], "PASS")
            self.assertEqual(results["broken"]["verdict"], "FAIL")
            run = tmp / "run"
            for name in ("01_logs.jsonl", "02_clusters.json", "02_selected.jsonl", "03_drafts.jsonl",
                         "03_review.html", "03_decisions.json", "03_eval_set.jsonl", "04_scores_baseline.jsonl",
                         "05_gate_broken.json", "report.md", "run_meta.json"):
                self.assertTrue((run / name).exists(), name)

            meta = read_json(run / "run_meta.json")["steps"]
            self.assertEqual(meta["ingest"]["records_rejected"], 3)
            self.assertEqual(meta["cluster"]["selected"], 16)
            self.assertGreater(meta["review"]["edit"], 0)  # the fake drafter always adds a fact
            summary = read_json(run / "04_summary_baseline.json")
            self.assertEqual(summary["action_accuracy"], 1.0)
            self.assertEqual(summary["det_score"], 1.0)
            self.assertEqual(len(read_jsonl(run / "03_eval_set.jsonl")), summary["items"])
            self.assertIn("Gate: broken vs baseline: **FAIL**", (run / "report.md").read_text())
            # one model in memory at a time: every model is released after its phase
            self.assertEqual(fake.released, ["drafter", "good-model", "good-model-2", "bad-model", "judge"])


if __name__ == "__main__":
    unittest.main()
