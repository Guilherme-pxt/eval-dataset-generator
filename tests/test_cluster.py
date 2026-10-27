import json
import unittest

import numpy as np

from evalgen.cluster import ClusterConfig, TfidfVectorizer, cluster_and_select, dedupe, purity, silhouette, spherical_kmeans
from evalgen.schema import LogRecord
from evalgen.synth.generate import generate
from evalgen.synth.kb import KB_DOCS
from evalgen.textutil import tokenize_for_clustering


def synthetic_records(n=300, seed=42):
    lines, truths = generate(n, seed=seed)
    records = []
    for line in lines:
        try:
            records.append(LogRecord.from_dict(json.loads(line)))
        except Exception:
            continue
    return records, {t["id"]: t for t in truths}


class TestPrimitives(unittest.TestCase):
    def test_tfidf_rows_are_unit_norm(self):
        docs = [tokenize_for_clustering(t) for t in ["growth plan price", "invoice INV-12345 status", "???"]]
        X = TfidfVectorizer().fit_transform(docs)
        np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0)

    def test_dedupe_groups_near_duplicates(self):
        texts = ["How much is the Growth plan?", "hi, how much is the growth plan? thanks",
                 "What is the API rate limit?", "Where is my data stored?"]
        X = TfidfVectorizer().fit_transform([tokenize_for_clustering(t) for t in texts])
        self.assertEqual(dedupe(X, 0.85), [[0, 1], [2], [3]])

    def test_kmeans_is_deterministic_and_separates_blobs(self):
        rng = np.random.default_rng(0)
        a = rng.normal([5, 0, 0], 0.3, size=(20, 3))
        b = rng.normal([0, 5, 0], 0.3, size=(20, 3))
        X = np.vstack([a, b])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        l1, _, _ = spherical_kmeans(X, 2, seed=3)
        l2, _, _ = spherical_kmeans(X, 2, seed=3)
        np.testing.assert_array_equal(l1, l2)
        self.assertEqual(len(set(l1[:20])), 1)
        self.assertEqual(len(set(l1[20:])), 1)
        self.assertNotEqual(l1[0], l1[-1])
        self.assertGreater(silhouette(X, l1), silhouette(X, rng.integers(0, 2, 40)))


class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.truth = synthetic_records()
        cls.cfg = ClusterConfig(budget=30, k_min=12, k_max=20)
        cls.items, cls.summary = cluster_and_select(cls.records, KB_DOCS, cls.cfg)

    def test_budget_and_determinism(self):
        self.assertEqual(len(self.items), 30)
        items2, _ = cluster_and_select(self.records, KB_DOCS, self.cfg)
        self.assertEqual([i.to_dict() for i in self.items], [i.to_dict() for i in items2])
        self.assertTrue(all(i.selection_reasons for i in self.items))

    def test_selection_is_enriched_for_failures(self):
        base = sum(1 for t in self.truth.values() if t["failure_type"]) / len(self.truth)
        sel = sum(1 for i in self.items if self.truth[i.log_id]["failure_type"]) / len(self.items)
        self.assertGreater(sel, 2 * base)

    def test_clusters_track_intents(self):
        p = purity(self.summary["assignments"], {k: v["intent"] for k, v in self.truth.items()})
        self.assertGreater(p, 0.6)  # regression guard; README reports the exact value


if __name__ == "__main__":
    unittest.main()
