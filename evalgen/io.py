"""JSON / JSONL helpers. Artifacts are plain files so every step is inspectable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield (line_number, raw_line) for non-blank lines."""
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip():
                yield n, line


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    for n, line in iter_jsonl(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    return n


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
