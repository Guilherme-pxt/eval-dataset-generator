"""Step 1: pull prompts and responses out of a production log.

Reads a JSONL log, validates every line against ``LogRecord``, drops what
cannot be trusted (and says why), redacts e-mail addresses and phone
numbers, applies optional filters and writes a clean JSONL file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .io import iter_jsonl, write_jsonl
from .schema import LogRecord, SchemaError
from .textutil import redact_pii


@dataclass
class IngestReport:
    read: int = 0
    kept: int = 0
    rejected: list[dict] = field(default_factory=list)
    filtered_out: int = 0
    pii_redactions: int = 0
    unknown_doc_ids: int = 0

    def to_dict(self) -> dict:
        return {
            "lines_read": self.read,
            "records_kept": self.kept,
            "records_rejected": len(self.rejected),
            "rejections": self.rejected,
            "filtered_out": self.filtered_out,
            "pii_redactions": self.pii_redactions,
            "unknown_doc_ids_dropped": self.unknown_doc_ids,
        }


def ingest(
    log_path: str | Path,
    kb: dict,
    out_path: str | Path | None = None,
    since: str | None = None,
    until: str | None = None,
    model_version: str | None = None,
    limit: int | None = None,
) -> tuple[list[LogRecord], IngestReport]:
    report = IngestReport()
    records: list[LogRecord] = []
    seen_ids: set[str] = set()
    for line_no, raw in iter_jsonl(log_path):
        report.read += 1
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            report.rejected.append({"line": line_no, "reason": f"invalid JSON: {exc.msg}"})
            continue
        try:
            rec = LogRecord.from_dict(data)
        except SchemaError as exc:
            report.rejected.append({"line": line_no, "reason": str(exc)})
            continue
        if rec.id in seen_ids:
            report.rejected.append({"line": line_no, "reason": f"duplicate id {rec.id}"})
            continue
        seen_ids.add(rec.id)

        # ISO-8601 UTC timestamps compare correctly as strings.
        if (since and rec.timestamp < since) or (until and rec.timestamp >= until):
            report.filtered_out += 1
            continue
        if model_version and rec.model_version != model_version:
            report.filtered_out += 1
            continue

        known = [d for d in rec.retrieved_doc_ids if d in kb]
        report.unknown_doc_ids += len(rec.retrieved_doc_ids) - len(known)
        rec.retrieved_doc_ids = known

        rec.user_message, n1 = redact_pii(rec.user_message)
        rec.response.message, n2 = redact_pii(rec.response.message)
        report.pii_redactions += n1 + n2

        records.append(rec)
        if limit and len(records) >= limit:
            break

    report.kept = len(records)
    if out_path:
        write_jsonl(out_path, (r.to_dict() for r in records))
    return records, report


def load_records(path: str | Path) -> list[LogRecord]:
    from .io import read_jsonl

    return [LogRecord.from_dict(d) for d in read_jsonl(path)]
