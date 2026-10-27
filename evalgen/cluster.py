"""Step 2: surface the cases that matter.

1. TF-IDF vectors (numpy) over the user messages.
2. Near-duplicate collapse: greedy grouping at cosine >= ``dedupe_threshold``.
3. Spherical k-means (k-means++ init, fixed seed, several restarts) over the
   group representatives, weighted by group size. k is picked by silhouette.
4. Selection under a budget: one representative per cluster, then the
   groups with the strongest failure signals, then prompts that are lexically
   isolated inside their cluster (likely rare intents), then coverage of the
   largest clusters.

Everything is deterministic for a given seed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .schema import EvalItem, LogRecord
from .signals import compute_signals, signal_score
from .textutil import tokenize_for_clustering


# ---------------------------------------------------------------- TF-IDF


def _features(tokens: list[str]) -> list[str]:
    feats = list(tokens)
    feats += [f"{a}__{b}" for a, b in zip(tokens, tokens[1:])]
    return feats or ["<empty>"]


class TfidfVectorizer:
    """Unigram + bigram TF-IDF with sublinear tf and L2 normalisation."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, docs: list[list[str]]) -> "TfidfVectorizer":
        df: Counter = Counter()
        for toks in docs:
            df.update(set(_features(toks)))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)])
        return self

    def transform(self, docs: list[list[str]]) -> np.ndarray:
        assert self.idf is not None, "fit first"
        X = np.zeros((len(docs), len(self.vocab)))
        for r, toks in enumerate(docs):
            for t, c in Counter(_features(toks)).items():
                j = self.vocab.get(t)
                if j is not None:
                    X[r, j] = 1.0 + math.log(c)
        X *= self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, docs: list[list[str]]) -> np.ndarray:
        return self.fit(docs).transform(docs)


# ---------------------------------------------------------------- dedupe


def dedupe(X: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy near-duplicate grouping in input order. Returns groups of row indices;
    the first index of each group is its representative."""
    groups: list[list[int]] = []
    reps: list[int] = []
    for i in range(X.shape[0]):
        if reps:
            sims = X[reps] @ X[i]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                groups[j].append(i)
                continue
        reps.append(i)
        groups.append([i])
    return groups


# ---------------------------------------------------------------- k-means


def _kmeans_pp(X: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    first = rng.choice(n, p=w / w.sum())
    centers = [X[first]]
    d2 = np.clip(1.0 - X @ X[first], 0, None)
    for _ in range(1, k):
        p = w * d2
        if p.sum() <= 0:
            idx = rng.choice(n)
        else:
            idx = rng.choice(n, p=p / p.sum())
        centers.append(X[idx])
        d2 = np.minimum(d2, np.clip(1.0 - X @ X[idx], 0, None))
    return np.array(centers)


def spherical_kmeans(
    X: np.ndarray, k: int, weights: np.ndarray | None = None, seed: int = 0, n_init: int = 8, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cosine k-means. Returns (labels, unit centroids, weighted inertia)."""
    n = X.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    k = min(k, n)
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        C = _kmeans_pp(X, w, k, rng)
        labels = np.full(n, -1)
        for _it in range(max_iter):
            new = np.argmax(X @ C.T, axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                mask = labels == c
                if not mask.any():  # re-seed an empty cluster at the worst-fit point
                    worst = int(np.argmin(np.max(X @ C.T, axis=1)))
                    C[c] = X[worst]
                    continue
                v = (X[mask] * w[mask, None]).sum(axis=0)
                nv = np.linalg.norm(v)
                C[c] = v / nv if nv > 0 else X[mask][0]
        inertia = float((w * (1.0 - np.max(X @ C.T, axis=1))).sum())
        if best is None or inertia < best[2] - 1e-12:
            best = (labels.copy(), C.copy(), inertia)
    return best


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette with cosine distance. Singletons score 0 (sklearn convention)."""
    D = np.clip(1.0 - X @ X.T, 0, None)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def choose_k(X, weights, k_min, k_max, seed) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in range(max(2, k_min), min(k_max, X.shape[0] - 1) + 1):
        labels, _, _ = spherical_kmeans(X, k, weights, seed=seed)
        scores[k] = round(silhouette(X, labels), 4)
    best_k = max(scores, key=lambda k: (scores[k], -k))
    return best_k, scores


# ---------------------------------------------------------------- pipeline step


@dataclass
class ClusterConfig:
    dedupe_threshold: float = 0.85
    k_min: int = 10
    k_max: int = 32
    k: int | None = None
    budget: int = 40
    max_failures_per_cluster: int = 2
    rare_cluster_size: int = 4
    failure_share: float = 0.3
    novelty_share: float = 0.2
    novelty_threshold: float = 0.35
    seed: int = 42


def cluster_and_select(records: list[LogRecord], kb: dict, cfg: ClusterConfig) -> tuple[list[EvalItem], dict]:
    msgs = [r.user_message for r in records]
    doc_texts = [f"{d['title']} {d['text']}" for d in kb.values()]
    toks = [tokenize_for_clustering(m) for m in msgs]
    doc_toks = [tokenize_for_clustering(t) for t in doc_texts]

    vec = TfidfVectorizer().fit(toks)
    X = vec.transform(toks)
    # A prompt that shares no content word with the documentation is an
    # off-topic candidate. Crude, but it has no tunable threshold.
    kb_vocab = {t for d in doc_toks for t in d}
    off_topic = np.array([not any(t in kb_vocab for t in tk) for tk in toks])

    signals = [compute_signals(r, kb, bool(off_topic[i])) for i, r in enumerate(records)]
    scores = np.array([signal_score(s) for s in signals])

    groups = dedupe(X, cfg.dedupe_threshold)
    rep_idx = np.array([g[0] for g in groups])
    sizes = np.array([len(g) for g in groups], dtype=float)
    Xr = X[rep_idx]

    if cfg.k:
        k, sil_scores = cfg.k, {}
    else:
        k, sil_scores = choose_k(Xr, sizes, cfg.k_min, cfg.k_max, cfg.seed)
    labels, C, inertia = spherical_kmeans(Xr, k, sizes, seed=cfg.seed)

    # Name clusters by their heaviest centroid terms (unigrams only, for readability).
    inv_vocab = {i: t for t, i in vec.vocab.items()}
    names = {}
    for c in range(C.shape[0]):
        order = np.argsort(-C[c])
        terms = [inv_vocab[j] for j in order if "__" not in inv_vocab[j]][:3]
        names[c] = " / ".join(terms)

    cluster_sizes = Counter()  # in log records, not groups
    for gi, g in enumerate(groups):
        cluster_sizes[int(labels[gi])] += len(g)

    # Per group: exemplar = member with the strongest signals (ties: earliest).
    group_info = []
    for gi, g in enumerate(groups):
        best = max(g, key=lambda i: (scores[i], -i))
        group_signals = sorted({s for i in g for s in signals[i]})
        group_info.append({
            "group": gi,
            "cluster": int(labels[gi]),
            "exemplar": best,
            "score": float(scores[g].max()),
            "n_flagged": int((scores[g] > 0).sum()),
            "size": len(g),
            "centrality": float(Xr[gi] @ C[labels[gi]]),
            "signals": group_signals,
        })

    selected: dict[int, list[str]] = {}  # group index -> reasons

    def add(gi: int, reason: str):
        selected.setdefault(gi, [])
        if reason not in selected[gi]:
            selected[gi].append(reason)

    # (1) one representative per cluster, largest clusters first
    by_cluster: dict[int, list[dict]] = {}
    for info in group_info:
        by_cluster.setdefault(info["cluster"], []).append(info)
    for c in sorted(by_cluster, key=lambda c: -cluster_sizes[c]):
        if len(selected) >= cfg.budget:
            break
        medoid = max(by_cluster[c], key=lambda g: (g["centrality"] * g["size"], -g["group"]))
        add(medoid["group"], "representative")
        if cluster_sizes[c] <= cfg.rare_cluster_size:
            add(medoid["group"], "rare_cluster")

    # (2) failure signals, strongest first, capped per cluster and by budget share
    fail_cap = len(selected) + int(round(cfg.failure_share * cfg.budget))
    per_cluster_fail: Counter = Counter()
    flagged = sorted((g for g in group_info if g["score"] > 0), key=lambda g: (-g["score"], -g["n_flagged"], g["group"]))
    for g in flagged:
        if g["group"] in selected:
            add(g["group"], "failure_signal")
            continue
        if len(selected) >= min(cfg.budget, fail_cap):
            continue
        if per_cluster_fail[g["cluster"]] >= cfg.max_failures_per_cluster:
            continue
        per_cluster_fail[g["cluster"]] += 1
        add(g["group"], "failure_signal")

    # (3) novelty: prompts that share little with the rest of their cluster. Rare
    # intents that k-means merged into a bigger cluster tend to land here.
    novelty_cap = len(selected) + int(round(cfg.novelty_share * cfg.budget))
    isolation = []
    for c, members in by_cluster.items():
        idx = [g["group"] for g in members]
        if len(idx) < 2:
            continue
        S = Xr[idx] @ Xr[idx].T
        np.fill_diagonal(S, -1.0)
        for pos, gi in enumerate(idx):
            isolation.append((float(S[pos].max()), gi))
    for nn_sim, gi in sorted(isolation):
        if len(selected) >= min(cfg.budget, novelty_cap) or nn_sim >= cfg.novelty_threshold:
            break
        if gi not in selected:
            add(gi, "novel_in_cluster")

    # (4) coverage: spend what is left on the largest clusters, least central groups first
    while len(selected) < min(cfg.budget, len(groups)):
        counts = Counter(group_info[gi]["cluster"] for gi in selected)
        candidates = [c for c in by_cluster if any(g["group"] not in selected for g in by_cluster[c])]
        if not candidates:
            break
        c = max(candidates, key=lambda c: (cluster_sizes[c] / (1 + counts[c]), -c))
        remaining = [g for g in by_cluster[c] if g["group"] not in selected]
        pick = min(remaining, key=lambda g: (g["centrality"], g["group"]))
        add(pick["group"], "coverage")

    items = []
    for gi in sorted(selected, key=lambda gi: group_info[gi]["exemplar"]):
        info = group_info[gi]
        rec = records[info["exemplar"]]
        items.append(EvalItem(
            item_id=f"item-{len(items) + 1:03d}",
            log_id=rec.id,
            user_message=rec.user_message,
            context_doc_ids=list(rec.retrieved_doc_ids),
            cluster_id=info["cluster"],
            cluster_label=names[info["cluster"]],
            selection_reasons=selected[gi],
            failure_signals=info["signals"],
            group_size=info["size"],
            observed_response=rec.response.to_dict(),
        ))

    clusters = []
    for c in sorted(by_cluster, key=lambda c: -cluster_sizes[c]):
        members = [records[i] for g in by_cluster[c] for i in groups[g["group"]]]
        clusters.append({
            "cluster_id": c,
            "label": names[c],
            "records": cluster_sizes[c],
            "unique_prompts": len(by_cluster[c]),
            "flagged_records": int(sum(1 for g in by_cluster[c] for i in groups[g["group"]] if scores[i] > 0)),
            "examples": [m.user_message for m in members[:3]],
        })

    summary = {
        "records": len(records),
        "near_duplicate_groups": len(groups),
        "records_collapsed_as_duplicates": len(records) - len(groups),
        "k": int(C.shape[0]),
        "silhouette_by_k": sil_scores,
        "inertia": round(inertia, 4),
        "off_topic_candidates": int(off_topic.sum()),
        "records_with_signals": int((scores > 0).sum()),
        "signal_counts": dict(Counter(s for sig in signals for s in sig)),
        "selected": len(items),
        "selection_reasons": dict(Counter(r for it in items for r in it.selection_reasons)),
        "clusters": clusters,
        "assignments": {records[i].id: int(labels[gi]) for gi, g in enumerate(groups) for i in g},
    }
    return items, summary


def purity(assignments: dict[str, int], truth: dict[str, str]) -> float:
    """Fraction of records whose cluster's majority intent equals their own intent."""
    by_cluster: dict[int, Counter] = {}
    for rid, c in assignments.items():
        if rid in truth:
            by_cluster.setdefault(c, Counter())[truth[rid]] += 1
    total = sum(sum(cnt.values()) for cnt in by_cluster.values())
    hit = sum(cnt.most_common(1)[0][1] for cnt in by_cluster.values())
    return hit / total if total else 0.0
