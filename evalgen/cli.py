"""Command line entry point: ``python -m evalgen <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline as P
from .assistant import ModelVersion
from .providers import ProviderError, build_provider


def _parse_version(spec: str) -> ModelVersion:
    """name=model[@prompt], e.g. cand2=qwen2.5:7b@v1-no-policy"""
    if "=" not in spec:
        raise argparse.ArgumentTypeError("expected name=model[@prompt]")
    name, rest = spec.split("=", 1)
    model, _, prompt = rest.partition("@")
    return ModelVersion(name=name, model=model, prompt=prompt or "v1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalgen", description=(
        "Turn production LLM logs into a reviewed eval set, score model versions against it, "
        "and gate releases on regressions."))
    p.add_argument("--config", default="config.toml", help="TOML config (default: config.toml)")
    p.add_argument("--run-dir", help="artifact directory (default: [run].run_dir)")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the LLM response cache")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate the synthetic production log (seeded)")
    s.add_argument("--records", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--data-dir")

    s = sub.add_parser("ingest", help="step 1: validate, filter and redact the production log")
    s.add_argument("--log")
    s.add_argument("--since", help="ISO timestamp, inclusive")
    s.add_argument("--until", help="ISO timestamp, exclusive")
    s.add_argument("--model-version")
    s.add_argument("--limit", type=int)

    s = sub.add_parser("cluster", help="step 2: dedupe, cluster and select cases to review")
    s.add_argument("--budget", type=int)
    s.add_argument("--k", type=int, help="fixed number of clusters (default: pick by silhouette)")

    s = sub.add_parser("draft", help="step 3a: draft expected outputs with an LLM")
    s.add_argument("--model")

    s = sub.add_parser("review", help="step 3b: review drafts (cli | html | oracle | apply)")
    s.add_argument("mode", choices=["cli", "html", "oracle", "apply"])
    s.add_argument("--decisions", help="decisions JSON (cli: where to save; apply: what to apply)")
    s.add_argument("--allow-partial", action="store_true", help="apply: ignore undecided items")

    s = sub.add_parser("score", help="step 4: run versions on the eval set and grade them")
    s.add_argument("--versions", nargs="+", help="version names from the config (default: all)")
    s.add_argument("--add-version", action="append", type=_parse_version, default=[],
                   metavar="NAME=MODEL[@PROMPT]", help="extra version not in the config")
    s.add_argument("--no-judge", action="store_true", help="deterministic checks only")

    s = sub.add_parser("gate", help="step 5: exit 1 if the candidate regresses vs the baseline")
    s.add_argument("--baseline")
    s.add_argument("--candidate")
    s.add_argument("--metric", choices=["combined", "det_score", "judge_norm"])
    s.add_argument("--max-overall-drop", type=float)
    s.add_argument("--max-cluster-drop", type=float)
    s.add_argument("--min-cluster-items", type=int)
    s.add_argument("--max-new-failures", type=int)
    s.add_argument("--out", help="gate result file name inside the run dir (default: 05_gate_<candidate>.json)")

    sub.add_parser("report", help="write report.md from the run artifacts")

    s = sub.add_parser("run-all", help="every step end to end; gates every version against the baseline")
    s.add_argument("--skip-synth", action="store_true", help="reuse the existing data/ files")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = P.load_config(args.config)
    run_dir = Path(args.run_dir or cfg["run"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    def provider():
        cache = None if args.no_cache else cfg["run"].get("cache")
        return build_provider(cfg["llm"], cache_path=cache)

    try:
        if args.command == "synth":
            if args.records:
                cfg["synth"]["records"] = args.records
            if args.seed is not None:
                cfg["run"]["seed"] = args.seed
            if args.data_dir:
                cfg["run"]["data_dir"] = args.data_dir
            P.step_synth(cfg)
        elif args.command == "ingest":
            icfg = cfg.setdefault("ingest", {})
            for k in ("log", "since", "until", "model_version", "limit"):
                if getattr(args, k) is not None:
                    icfg[k] = getattr(args, k)
            P.step_ingest(cfg, run_dir)
        elif args.command == "cluster":
            if args.budget:
                cfg["cluster"]["budget"] = args.budget
            if args.k:
                cfg["cluster"]["k"] = args.k
            P.step_cluster(cfg, run_dir)
        elif args.command == "draft":
            if args.model:
                cfg["draft"]["model"] = args.model
            P.step_draft(cfg, run_dir, provider())
        elif args.command == "review":
            from .review import render_review_html, review_cli

            if args.mode == "html":
                out = run_dir / P.F["review_html"]
                out.write_text(render_review_html(P._items(run_dir / P.F["drafts"]), P.load_kb(cfg)), encoding="utf-8")
                print(f"open {out} in a browser, review, then: python -m evalgen review apply --decisions <exported file>")
            elif args.mode == "cli":
                path = Path(args.decisions or run_dir / P.F["decisions"])
                review_cli(P._items(run_dir / P.F["drafts"]), path, P.load_kb(cfg))
                print(f"saved to {path}; next: python -m evalgen review apply --decisions {path}")
            elif args.mode == "oracle":
                decisions = P.step_review_oracle(cfg, run_dir)
                print(f"oracle wrote {len(decisions)} decisions to {run_dir / P.F['decisions']}")
            else:
                P.step_review_apply(cfg, run_dir, decisions_path=args.decisions, allow_partial=args.allow_partial)
        elif args.command == "score":
            extra = {v.name: v for v in args.add_version}
            names = (args.versions or list(P.versions_from_config(cfg))) + [n for n in extra if n not in (args.versions or [])]
            P.step_score(cfg, run_dir, provider(), version_names=names, use_judge=not args.no_judge,
                         extra_versions=extra)
        elif args.command == "gate":
            result = P.step_gate(cfg, run_dir, baseline=args.baseline, candidate=args.candidate, overrides={
                "metric": args.metric, "max_overall_drop": args.max_overall_drop,
                "max_cluster_drop": args.max_cluster_drop, "min_cluster_items": args.min_cluster_items,
                "max_new_failures": args.max_new_failures,
            }, out_name=args.out)
            return 1 if result["verdict"] == "FAIL" else 0
        elif args.command == "report":
            P.step_report(cfg, run_dir)
        elif args.command == "run-all":
            results = P.run_all(cfg, run_dir, provider(), skip_synth=args.skip_synth)
            return 1 if any(r["verdict"] == "FAIL" for r in results.values()) else 0
    except (ProviderError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
