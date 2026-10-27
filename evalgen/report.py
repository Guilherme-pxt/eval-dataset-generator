"""Markdown report built only from run artifacts (summaries, gate results, run metadata)."""

from __future__ import annotations


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_report(summaries: dict, gates: list[dict], meta: dict, clusters: dict) -> str:
    out = ["# Eval run report", ""]
    steps = meta.get("steps", {})

    if clusters:
        out += ["## Dataset", ""]
        ing = steps.get("ingest", {})
        rev = steps.get("review", {})
        out.append(f"- Log lines read: {ing.get('lines_read', 'n/a')}, records kept: {ing.get('records_kept', 'n/a')}, "
                   f"rejected: {ing.get('records_rejected', 'n/a')}, PII spans redacted: {ing.get('pii_redactions', 'n/a')}")
        out.append(f"- Unique prompts after near-duplicate collapse: {clusters.get('near_duplicate_groups')}; "
                   f"clusters: {clusters.get('k')}; selected for review: {clusters.get('selected')} "
                   f"{clusters.get('selection_reasons')}")
        diag = clusters.get("diagnostics_vs_ground_truth")
        if diag:
            out.append(f"- Against the synthetic ground truth: cluster purity {diag['cluster_purity_vs_intent']}, "
                       f"intents covered {diag['intents_covered_by_selection']}/{diag['intents_in_log']} "
                       f"(rare: {diag['rare_intents_covered']}), production failure rate "
                       f"{diag['failure_rate_in_log']:.1%} in the log vs {diag['failure_rate_in_selection']:.1%} in the selection")
        if rev:
            out.append(f"- Review ({', '.join(rev.get('reviewers', []))}): accepted {rev.get('accept')}, edited {rev.get('edit')}, "
                       f"rejected {rev.get('reject')} -> {rev.get('eval_items')} eval items; "
                       f"draft acceptance rate {_f(rev.get('draft_acceptance_rate'))}")
        out.append("")

    if summaries:
        out += ["## Overall", "",
                "| version | model | prompt | items | valid JSON | action acc. | det score | det pass | judge (1-5) | combined | judge/det agreement | mean latency (s) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in summaries.items():
            out.append(f"| {name} | {s['model']} | {s['prompt']} | {s['items']} | {_f(s['valid_json_rate'])} | "
                       f"{_f(s['action_accuracy'])} | {_f(s['det_score'])} | {_f(s['det_pass_rate'])} | "
                       f"{_f(s['judge_mean_1to5'], 2)} | {_f(s['combined'])} | {_f(s['judge_det_agreement'])} | "
                       f"{_f(s['mean_latency_s'], 2)} |")
        out.append("")
        out += ["### By expected action", "", "| version | " + " | ".join(
            f"{a} (n) det / combined" for a in ("answer", "tool_call", "decline")) + " |", "|---|---|---|---|"]
        for name, s in summaries.items():
            cells = []
            for a in ("answer", "tool_call", "decline"):
                b = s["by_expected_action"].get(a)
                cells.append(f"({b['n']}) {_f(b['det_score'])} / {_f(b['combined'])}" if b else "n/a")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")

    for g in gates:
        out += [f"## Gate: {g['candidate']} vs {g['baseline']}: **{g['verdict']}**", ""]
        out.append(f"- {g['metric']}: {g['baseline_mean']} -> {g['candidate_mean']} "
                   f"(delta {g['delta']:+.3f}, paired bootstrap 95% CI [{g['delta_ci95'][0]:+.3f}, {g['delta_ci95'][1]:+.3f}])")
        out.append(f"- items newly failing: {len(g['new_failures'])}, items fixed: {len(g['fixed'])}")
        for r in g["reasons"]:
            out.append(f"- reason: {r}")
        out += ["", "| cluster | n | baseline | candidate | delta | gated | regressed |", "|---|---|---|---|---|---|---|"]
        for pc in sorted(g["per_cluster"], key=lambda p: p["delta"]):
            out.append(f"| {pc['label']} | {pc['n']} | {_f(pc['baseline'])} | {_f(pc['candidate'])} | "
                       f"{pc['delta']:+.3f} | {'yes' if pc['gated'] else 'no'} | {'YES' if pc['regressed'] else ''} |")
        out.append("")

    if steps:
        out += ["## Timing", "", "| step | wall time (s) |", "|---|---|"]
        for name, info in steps.items():
            if name == "score":
                for v, vi in info.items():
                    out.append(f"| score: {v} predict | {vi.get('predict_wall_s')} |")
                    out.append(f"| score: {v} judge | {vi.get('judge_wall_s')} |")
            elif isinstance(info, dict) and "wall_s" in info:
                out.append(f"| {name} | {info['wall_s']} |")
        out.append("")
    return "\n".join(out)
