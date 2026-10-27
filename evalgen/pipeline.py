"""Step functions shared by the CLI and ``run-all``.

Each step reads the previous step's artifact from the run directory and
writes its own, so any step can be re-run in isolation.
"""

from __future__ import annotations

import platform
import time
import tomllib
from pathlib import Path

from .assistant import ModelVersion
from .cluster import ClusterConfig, cluster_and_select, purity
from .draft import draft_all
from .gate import GateConfig, run_gate
from .ingest import ingest, load_records
from .io import read_json, read_jsonl, write_json, write_jsonl
from .report import render_report
from .review import apply_decisions, load_decisions, oracle_decisions, render_review_html
from .schema import EvalItem, ReviewDecision
from .score import predict, score_version, summarize
from .synth.generate import write_dataset

F = {
    "logs": "01_logs.jsonl",
    "ingest": "01_ingest_report.json",
    "clusters": "02_clusters.json",
    "selected": "02_selected.jsonl",
    "drafts": "03_drafts.jsonl",
    "review_html": "03_review.html",
    "decisions": "03_decisions.json",
    "eval_set": "03_eval_set.jsonl",
    "report": "report.md",
    "meta": "run_meta.json",
}


def preds_file(v: str) -> str:
    return f"04_predictions_{v}.jsonl"


def scores_file(v: str) -> str:
    return f"04_scores_{v}.jsonl"


def summary_file(v: str) -> str:
    return f"04_summary_{v}.json"


def gate_file(candidate: str) -> str:
    return f"05_gate_{candidate}.json"


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def versions_from_config(cfg: dict) -> dict[str, ModelVersion]:
    return {name: ModelVersion(name=name, model=v["model"], prompt=v.get("prompt", "v1"))
            for name, v in cfg.get("versions", {}).items()}


def load_kb(cfg: dict) -> dict:
    return read_json(Path(cfg["run"]["data_dir"]) / "kb.json")


def _meta_update(run_dir: Path, step: str, info: dict) -> None:
    path = run_dir / F["meta"]
    meta = read_json(path) if path.exists() else {"steps": {}}
    meta["steps"][step] = info
    meta["python"] = platform.python_version()
    write_json(path, meta)


def _items(path: Path) -> list[EvalItem]:
    return [EvalItem.from_dict(d) for d in read_jsonl(path)]


# ------------------------------------------------------------------ steps


def step_synth(cfg: dict, log=print) -> dict:
    t0 = time.perf_counter()
    info = write_dataset(cfg["run"]["data_dir"], n_records=int(cfg["synth"]["records"]), seed=int(cfg["run"]["seed"]))
    info["wall_s"] = round(time.perf_counter() - t0, 2)
    log(f"synth: {info['records']} records, {info['malformed_lines']} malformed lines, failures {info['failure_counts']}")
    return info


def step_ingest(cfg: dict, run_dir: Path, log=print) -> dict:
    t0 = time.perf_counter()
    icfg = cfg.get("ingest", {})
    _, report = ingest(
        icfg.get("log", Path(cfg["run"]["data_dir"]) / "production_log.jsonl"), load_kb(cfg),
        out_path=run_dir / F["logs"], since=icfg.get("since"), until=icfg.get("until"),
        model_version=icfg.get("model_version"), limit=icfg.get("limit"),
    )
    info = report.to_dict()
    write_json(run_dir / F["ingest"], info)
    info = {k: v for k, v in info.items() if k != "rejections"}
    info["wall_s"] = round(time.perf_counter() - t0, 2)
    _meta_update(run_dir, "ingest", info)
    log(f"ingest: kept {info['records_kept']} of {info['lines_read']} lines, rejected {info['records_rejected']}, "
        f"redacted {info['pii_redactions']} PII spans")
    return info


def step_cluster(cfg: dict, run_dir: Path, log=print) -> dict:
    t0 = time.perf_counter()
    c = cfg.get("cluster", {})
    ccfg = ClusterConfig(
        dedupe_threshold=float(c.get("dedupe_threshold", 0.85)), k_min=int(c.get("k_min", 12)),
        k_max=int(c.get("k_max", 32)), k=c.get("k"), budget=int(c.get("budget", 40)),
        max_failures_per_cluster=int(c.get("max_failures_per_cluster", 2)),
        rare_cluster_size=int(c.get("rare_cluster_size", 4)), seed=int(cfg["run"]["seed"]),
    )
    records = load_records(run_dir / F["logs"])
    items, summary = cluster_and_select(records, load_kb(cfg), ccfg)

    # Diagnostics only possible because the data is synthetic: never used for selection.
    truth_path = Path(cfg["run"]["data_dir"]) / "ground_truth.jsonl"
    if truth_path.exists():
        truth = {t["id"]: t for t in read_jsonl(truth_path)}
        summary["diagnostics_vs_ground_truth"] = _selection_diagnostics(records, items, summary, truth)

    write_json(run_dir / F["clusters"], summary)
    write_jsonl(run_dir / F["selected"], (it.to_dict() for it in items))
    info = {k: summary[k] for k in ("records", "near_duplicate_groups", "k", "selected", "selection_reasons",
                                    "records_with_signals")}
    if "diagnostics_vs_ground_truth" in summary:
        info["diagnostics_vs_ground_truth"] = summary["diagnostics_vs_ground_truth"]
    info["wall_s"] = round(time.perf_counter() - t0, 2)
    _meta_update(run_dir, "cluster", info)
    log(f"cluster: {info['records']} records -> {info['near_duplicate_groups']} unique prompts -> k={info['k']} "
        f"clusters -> {info['selected']} selected {info['selection_reasons']}")
    return info


def _selection_diagnostics(records, items, summary, truth) -> dict:
    failing = [r.id for r in records if truth[r.id]["failure_type"]]
    flagged_failing = sum(1 for it in items if truth[it.log_id]["failure_type"])
    intents_all = {truth[r.id]["intent"] for r in records}
    intents_sel = {truth[it.log_id]["intent"] for it in items}
    rare = {i for i in intents_all if sum(1 for r in records if truth[r.id]["intent"] == i) <= 6}
    random_rate = len(failing) / len(records)
    return {
        "cluster_purity_vs_intent": round(purity(summary["assignments"], {k: v["intent"] for k, v in truth.items()}), 4),
        "intents_in_log": len(intents_all),
        "intents_covered_by_selection": len(intents_sel),
        "rare_intents_covered": f"{len(rare & intents_sel)}/{len(rare)}",
        "failing_records_in_log": len(failing),
        "failure_rate_in_log": round(random_rate, 4),
        "selected_items_that_failed_in_production": flagged_failing,
        "failure_rate_in_selection": round(flagged_failing / len(items), 4) if items else 0.0,
    }


def step_draft(cfg: dict, run_dir: Path, provider, log=print) -> dict:
    items = _items(run_dir / F["selected"])
    model = cfg["draft"]["model"]
    log(f"draft: {len(items)} items with {model}")
    items, info = draft_all(items, provider, model, load_kb(cfg), log=log)
    write_jsonl(run_dir / F["drafts"], (it.to_dict() for it in items))
    (run_dir / F["review_html"]).write_text(render_review_html(items, load_kb(cfg)), encoding="utf-8")
    _meta_update(run_dir, "draft", info)
    log(f"draft: done in {info['wall_s']}s, {info['failed']} failed, {info['with_warnings']} with warnings; "
        f"review page: {run_dir / F['review_html']}")
    return info


def step_review_oracle(cfg: dict, run_dir: Path, log=print) -> list[ReviewDecision]:
    truth = {t["id"]: t for t in read_jsonl(Path(cfg["run"]["data_dir"]) / "ground_truth.jsonl")}
    decisions = oracle_decisions(_items(run_dir / F["drafts"]), truth)
    write_json(run_dir / F["decisions"], [d.to_dict() for d in decisions])
    return decisions


def step_review_apply(cfg: dict, run_dir: Path, decisions_path: Path | None = None, allow_partial=False,
                      log=print) -> dict:
    t0 = time.perf_counter()
    decisions_path = Path(decisions_path or run_dir / F["decisions"])
    drafts = _items(run_dir / F["drafts"])
    decisions = load_decisions(decisions_path)
    eval_set, stats = apply_decisions(drafts, decisions, allow_partial=allow_partial)
    write_jsonl(run_dir / F["eval_set"], (it.to_dict() for it in eval_set))
    reviewers = sorted({d.reviewer for d in decisions})
    info = {**stats, "eval_items": len(eval_set), "reviewers": reviewers, "decisions_file": str(decisions_path),
            "draft_acceptance_rate": round(stats["accept"] / max(1, stats["accept"] + stats["edit"]), 4),
            "wall_s": round(time.perf_counter() - t0, 2)}
    _meta_update(run_dir, "review", info)
    log(f"review: accepted {stats['accept']}, edited {stats['edit']}, rejected {stats['reject']} "
        f"-> {len(eval_set)} eval items (reviewer: {', '.join(reviewers)})")
    return info


def step_score(cfg: dict, run_dir: Path, provider, version_names: list[str] | None = None, use_judge=True,
               extra_versions: dict[str, ModelVersion] | None = None, log=print) -> dict:
    """Predict with every version first, then judge everything: one model in memory at a time."""
    versions = versions_from_config(cfg)
    versions.update(extra_versions or {})
    names = version_names or list(versions)
    items = _items(run_dir / F["eval_set"])
    kb = load_kb(cfg)
    info = {}
    preds = {}
    for name in names:
        v = versions[name]
        t0 = time.perf_counter()
        log(f"score: predicting {len(items)} items with {name} ({v.model}, prompt {v.prompt})")
        preds[name] = predict(items, v, provider, kb, log=log)
        write_jsonl(run_dir / preds_file(name), preds[name])
        info[name] = {"model": v.model, "prompt": v.prompt, "predict_wall_s": round(time.perf_counter() - t0, 1)}

    judge_model = cfg["judge"]["model"] if use_judge else None
    for name in names:
        t0 = time.perf_counter()
        scores = score_version(items, preds[name], kb, judge=provider if use_judge else None,
                               judge_model=judge_model, log=log)
        write_jsonl(run_dir / scores_file(name), (s.to_dict() for s in scores))
        summary = summarize(scores)
        summary.update({"version": name, "model": versions[name].model, "prompt": versions[name].prompt,
                        "judge_model": judge_model})
        write_json(run_dir / summary_file(name), summary)
        info[name]["judge_wall_s"] = round(time.perf_counter() - t0, 1)
        info[name]["combined"] = summary["combined"]
        info[name]["det_score"] = summary["det_score"]
        log(f"score: {name} det={summary['det_score']} judge={summary['judge_mean_1to5']} "
            f"combined={summary['combined']}")
    if use_judge:
        provider.release(judge_model)
    _meta_update(run_dir, "score", info)
    return info


def step_gate(cfg: dict, run_dir: Path, baseline: str | None = None, candidate: str | None = None,
              overrides: dict | None = None, out_name: str | None = None, log=print) -> dict:
    g = dict(cfg.get("gate", {}))
    g.update({k: v for k, v in (overrides or {}).items() if v is not None})
    baseline = baseline or g.get("baseline", "baseline")
    candidate = candidate or g.get("candidate", "candidate")
    gcfg = GateConfig(
        metric=g.get("metric", "combined"), max_overall_drop=float(g.get("max_overall_drop", 0.05)),
        max_cluster_drop=float(g.get("max_cluster_drop", 0.15)), min_cluster_items=int(g.get("min_cluster_items", 3)),
        max_new_failures=int(g.get("max_new_failures", 3)), seed=int(cfg["run"]["seed"]),
    )
    result = run_gate(read_jsonl(run_dir / scores_file(baseline)), read_jsonl(run_dir / scores_file(candidate)), gcfg)
    result["baseline"], result["candidate"] = baseline, candidate
    write_json(run_dir / (out_name or gate_file(candidate)), result)
    log(f"gate: {result['verdict']} {candidate} vs {baseline}: {gcfg.metric} {result['baseline_mean']} -> "
        f"{result['candidate_mean']} (delta {result['delta']:+.3f}, 95% CI {result['delta_ci95']})")
    for r in result["reasons"]:
        log(f"  - {r}")
    return result


def step_report(cfg: dict, run_dir: Path, log=print) -> Path:
    summaries = {}
    for p in sorted(run_dir.glob("04_summary_*.json")):
        s = read_json(p)
        summaries[s["version"]] = s
    gates = [read_json(p) for p in sorted(run_dir.glob("05_gate*.json"))]
    meta = read_json(run_dir / F["meta"]) if (run_dir / F["meta"]).exists() else {}
    clusters = read_json(run_dir / F["clusters"]) if (run_dir / F["clusters"]).exists() else {}
    md = render_report(summaries, gates, meta, clusters)
    out = run_dir / F["report"]
    out.write_text(md, encoding="utf-8")
    log(f"report: {out}")
    return out


def run_all(cfg: dict, run_dir: Path, provider, skip_synth=False, log=print) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if not skip_synth:
        step_synth(cfg, log=log)
    step_ingest(cfg, run_dir, log=log)
    step_cluster(cfg, run_dir, log=log)
    step_draft(cfg, run_dir, provider, log=log)
    mode = cfg.get("review", {}).get("mode", "oracle")
    if mode == "oracle":
        step_review_oracle(cfg, run_dir, log=log)
        step_review_apply(cfg, run_dir, log=log)
    else:
        step_review_apply(cfg, run_dir, decisions_path=Path(cfg["review"]["decisions"]), log=log)
    step_score(cfg, run_dir, provider, log=log)
    baseline = cfg.get("gate", {}).get("baseline", "baseline")
    results = {name: step_gate(cfg, run_dir, baseline=baseline, candidate=name, log=log)
               for name in versions_from_config(cfg) if name != baseline}
    wall = round(time.perf_counter() - t0, 1)
    _meta_update(run_dir, "run_all", {"wall_s": wall})
    step_report(cfg, run_dir, log=log)
    log(f"run-all: finished in {wall}s")
    return results
