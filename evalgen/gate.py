"""Step 5: regression gate.

Compares a candidate's per-item scores with the baseline's on the same
reviewed eval set and fails (exit code 1) when:

* the overall mean of ``metric`` drops by more than ``max_overall_drop``, or
* any cluster with at least ``min_cluster_items`` items drops by more than
  ``max_cluster_drop``, or
* more than ``max_new_failures`` items pass on the baseline and fail on the candidate.

A paired bootstrap 95% interval of the overall delta is reported so the
reader can see how much of a small delta is noise. It is informational:
the thresholds decide, because a gate must be predictable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateConfig:
    metric: str = "combined"
    max_overall_drop: float = 0.05
    max_cluster_drop: float = 0.15
    min_cluster_items: int = 3
    max_new_failures: int = 3
    bootstrap_samples: int = 2000
    seed: int = 42


def _passed(row: dict) -> bool:
    return bool(row.get("det_pass")) and (row.get("judge_score") is None or row["judge_score"] >= 4)


def _metric(row: dict, metric: str) -> float:
    v = row.get(metric)
    if v is None:  # judge failed for this item: fall back to the deterministic score
        v = row.get("det_score", 0.0)
    return float(v)


def run_gate(baseline: list[dict], candidate: list[dict], cfg: GateConfig) -> dict:
    b = {r["item_id"]: r for r in baseline}
    c = {r["item_id"]: r for r in candidate}
    common = sorted(set(b) & set(c))
    if not common:
        raise ValueError("baseline and candidate share no items")
    if len(common) != len(b) or len(common) != len(c):
        raise ValueError("baseline and candidate were scored on different item sets")

    xb = np.array([_metric(b[i], cfg.metric) for i in common])
    xc = np.array([_metric(c[i], cfg.metric) for i in common])
    delta = float(xc.mean() - xb.mean())

    rng = np.random.default_rng(cfg.seed)
    idx = rng.integers(0, len(common), size=(cfg.bootstrap_samples, len(common)))
    boot = (xc[idx] - xb[idx]).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    clusters = {}
    for i in common:
        key = b[i]["cluster_id"]
        cl = clusters.setdefault(key, {"label": b[i]["cluster_label"], "b": [], "c": []})
        cl["b"].append(_metric(b[i], cfg.metric))
        cl["c"].append(_metric(c[i], cfg.metric))
    per_cluster = []
    for cid, cl in sorted(clusters.items()):
        n = len(cl["b"])
        d = float(np.mean(cl["c"]) - np.mean(cl["b"]))
        gated = n >= cfg.min_cluster_items
        per_cluster.append({
            "cluster_id": cid, "label": cl["label"], "n": n,
            "baseline": round(float(np.mean(cl["b"])), 4), "candidate": round(float(np.mean(cl["c"])), 4),
            "delta": round(d, 4), "gated": gated, "regressed": gated and d < -cfg.max_cluster_drop,
        })

    new_failures = [i for i in common if _passed(b[i]) and not _passed(c[i])]
    fixed = [i for i in common if not _passed(b[i]) and _passed(c[i])]

    reasons = []
    if delta < -cfg.max_overall_drop:
        reasons.append(f"overall {cfg.metric} dropped by {-delta:.3f} (limit {cfg.max_overall_drop})")
    for pc in per_cluster:
        if pc["regressed"]:
            reasons.append(f"cluster '{pc['label']}' (n={pc['n']}) dropped by {-pc['delta']:.3f} "
                           f"(limit {cfg.max_cluster_drop})")
    if len(new_failures) > cfg.max_new_failures:
        reasons.append(f"{len(new_failures)} items newly failing (limit {cfg.max_new_failures})")

    return {
        "verdict": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "metric": cfg.metric,
        "items": len(common),
        "baseline_mean": round(float(xb.mean()), 4),
        "candidate_mean": round(float(xc.mean()), 4),
        "delta": round(delta, 4),
        "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "new_failures": new_failures,
        "fixed": fixed,
        "per_cluster": per_cluster,
        "config": cfg.__dict__,
    }
