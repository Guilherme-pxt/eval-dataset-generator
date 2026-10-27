"""Record types for every pipeline stage, validated with plain dataclasses.

Each ``from_dict`` raises ``SchemaError`` with a precise message on bad
input. Validation lives at the boundaries (reading files, parsing model
output, applying review decisions); code in between trusts these types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ACTIONS = ("answer", "tool_call", "decline")
TOOLS = ("lookup_invoice", "create_refund_request", "escalate_to_human")
DECISIONS = ("accept", "edit", "reject")
FEEDBACK = ("up", "down", None)


class SchemaError(ValueError):
    pass


def _req(d: dict, key: str, typ, where: str):
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: expected an object, got {type(d).__name__}")
    if key not in d:
        raise SchemaError(f"{where}: missing field '{key}'")
    val = d[key]
    if not isinstance(val, typ):
        names = typ.__name__ if isinstance(typ, type) else "/".join(t.__name__ for t in typ)
        raise SchemaError(f"{where}: field '{key}' must be {names}, got {type(val).__name__}")
    return val


def _opt(d: dict, key: str, typ, where: str, default=None):
    if d.get(key) is None:
        return default
    return _req(d, key, typ, where)


def _str_list(d: dict, key: str, where: str, required: bool = True) -> list[str]:
    val = _req(d, key, list, where) if required else (_opt(d, key, list, where) or [])
    for i, x in enumerate(val):
        if not isinstance(x, str):
            raise SchemaError(f"{where}: {key}[{i}] must be a string")
    return list(val)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any, where: str = "tool") -> "ToolCall | None":
        if d is None:
            return None
        name = _req(d, "name", str, where)
        if name not in TOOLS:
            raise SchemaError(f"{where}: unknown tool '{name}' (known: {', '.join(TOOLS)})")
        args = _opt(d, "arguments", dict, where, default={})
        return cls(name=name, arguments={str(k): v for k, v in args.items()})


@dataclass
class AssistantResponse:
    action: str
    message: str
    tool: ToolCall | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str = "response") -> "AssistantResponse":
        action = _req(d, "action", str, where)
        if action not in ACTIONS:
            raise SchemaError(f"{where}: action must be one of {ACTIONS}, got '{action}'")
        message = _opt(d, "message", str, where, default="")
        tool = ToolCall.from_dict(d.get("tool"), where=f"{where}.tool")
        if action == "tool_call" and tool is None:
            raise SchemaError(f"{where}: action 'tool_call' requires a tool")
        if action != "tool_call":
            tool = None
        return cls(action=action, message=message, tool=tool)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogRecord:
    id: str
    timestamp: str
    model_version: str
    user_message: str
    retrieved_doc_ids: list[str]
    response: AssistantResponse
    feedback: str | None = None
    latency_ms: int | None = None
    session_id: str | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str = "log") -> "LogRecord":
        rid = _req(d, "id", str, where)
        where = f"{where}[{rid}]"
        feedback = d.get("feedback")
        if feedback not in FEEDBACK:
            raise SchemaError(f"{where}: feedback must be 'up', 'down' or null")
        latency = d.get("latency_ms")
        if latency is not None and not isinstance(latency, int):
            raise SchemaError(f"{where}: latency_ms must be an integer")
        user_message = _req(d, "user_message", str, where)
        if not user_message.strip():
            raise SchemaError(f"{where}: user_message is empty")
        return cls(
            id=rid,
            timestamp=_req(d, "timestamp", str, where),
            model_version=_req(d, "model_version", str, where),
            user_message=user_message,
            retrieved_doc_ids=_str_list(d, "retrieved_doc_ids", where),
            response=AssistantResponse.from_dict(_req(d, "response", dict, where), f"{where}.response"),
            feedback=feedback,
            latency_ms=latency,
            session_id=_opt(d, "session_id", str, where),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Expected:
    """What a correct assistant turn looks like for one eval item."""

    action: str
    tool: ToolCall | None
    required_facts: list[str]
    reference_answer: str

    @classmethod
    def from_dict(cls, d: Any, where: str = "expected") -> "Expected":
        action = _req(d, "action", str, where)
        if action not in ACTIONS:
            raise SchemaError(f"{where}: action must be one of {ACTIONS}, got '{action}'")
        tool = ToolCall.from_dict(d.get("tool"), f"{where}.tool")
        if action == "tool_call" and tool is None:
            raise SchemaError(f"{where}: action 'tool_call' requires a tool")
        if action != "tool_call":
            tool = None
        facts = [f.strip() for f in _str_list(d, "required_facts", where, required=False) if f.strip()]
        if action != "answer":
            facts = []
        return cls(
            action=action,
            tool=tool,
            required_facts=facts,
            reference_answer=_opt(d, "reference_answer", str, where, default=""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalItem:
    """A candidate or reviewed evaluation case.

    ``expected`` is None for a selected-but-not-yet-drafted item.
    """

    item_id: str
    log_id: str
    user_message: str
    context_doc_ids: list[str]
    cluster_id: int
    cluster_label: str
    selection_reasons: list[str]
    failure_signals: list[str]
    group_size: int
    observed_response: dict | None = None
    expected: Expected | None = None
    draft_warnings: list[str] = field(default_factory=list)
    review: dict | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str = "item") -> "EvalItem":
        iid = _req(d, "item_id", str, where)
        where = f"{where}[{iid}]"
        exp = d.get("expected")
        return cls(
            item_id=iid,
            log_id=_req(d, "log_id", str, where),
            user_message=_req(d, "user_message", str, where),
            context_doc_ids=_str_list(d, "context_doc_ids", where),
            cluster_id=_req(d, "cluster_id", int, where),
            cluster_label=_req(d, "cluster_label", str, where),
            selection_reasons=_str_list(d, "selection_reasons", where),
            failure_signals=_str_list(d, "failure_signals", where, required=False),
            group_size=_req(d, "group_size", int, where),
            observed_response=_opt(d, "observed_response", dict, where),
            expected=Expected.from_dict(exp, f"{where}.expected") if exp is not None else None,
            draft_warnings=_str_list(d, "draft_warnings", where, required=False),
            review=_opt(d, "review", dict, where),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewDecision:
    item_id: str
    decision: str
    expected: Expected | None = None
    note: str = ""
    reviewer: str = ""

    @classmethod
    def from_dict(cls, d: Any, where: str = "decision") -> "ReviewDecision":
        iid = _req(d, "item_id", str, where)
        where = f"{where}[{iid}]"
        decision = _req(d, "decision", str, where)
        if decision not in DECISIONS:
            raise SchemaError(f"{where}: decision must be one of {DECISIONS}, got '{decision}'")
        exp = d.get("expected")
        if decision == "edit" and exp is None:
            raise SchemaError(f"{where}: an 'edit' decision must carry the edited 'expected' block")
        return cls(
            item_id=iid,
            decision=decision,
            expected=Expected.from_dict(exp, f"{where}.expected") if exp is not None else None,
            note=_opt(d, "note", str, where, default=""),
            reviewer=_opt(d, "reviewer", str, where, default=""),
        )

    def to_dict(self) -> dict:
        return asdict(self)
