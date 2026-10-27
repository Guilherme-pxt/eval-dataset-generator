"""The assistant under test: prompt variants, output schema and parsing.

A "model version" is the pair (model, prompt variant). Changing either one is
a release that the gate should vet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ACTIONS, TOOLS, AssistantResponse, SchemaError
from .synth.kb import context_text
from .textutil import parse_json_loose

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "tool": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(TOOLS)},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            ]
        },
        "message": {"type": "string"},
    },
    "required": ["action", "tool", "message"],
}

TOOL_DOCS = """Tools:
- lookup_invoice(invoice_id): status of one specific invoice, e.g. INV-12345.
- create_refund_request(account_id, reason): the user asks for a refund on a specific account, e.g. ACC-12345.
- escalate_to_human(reason): payment disputes and chargebacks."""

OUTPUT_FORMAT = """Reply with one JSON object:
{"action": "answer" | "tool_call" | "decline", "tool": {"name": "...", "arguments": {...}} or null, "message": "text shown to the user"}"""

POLICY = """Rules:
1. Answer only from the documentation below. Never invent prices, limits, dates or other numbers. If the documentation does not cover the question, say so and suggest contacting support.
2. If the request is unrelated to Ledgerfox, set "action" to "decline" and say politely that you can only help with Ledgerfox.
3. If the request needs account-specific data or an action, set "action" to "tool_call" and fill "tool":
   lookup_invoice for a specific invoice, create_refund_request for a refund on a specific account, escalate_to_human for payment disputes and chargebacks.
4. Otherwise set "action" to "answer" and keep "tool" null. Be concise."""

PROMPTS = {
    # the production prompt
    "v1": "You are the customer support assistant for Ledgerfox, an invoicing app.\n\n"
          + POLICY + "\n\n" + TOOL_DOCS + "\n\n" + OUTPUT_FORMAT + "\n\nDocumentation:\n{context}",
    # a "simplified" prompt that drops the policy: a realistic regression to catch
    "v1-no-policy": "You are a friendly, helpful assistant for Ledgerfox, an invoicing app.\n\n"
          + TOOL_DOCS + "\n\n" + OUTPUT_FORMAT + "\n\nDocumentation:\n{context}",
}


@dataclass(frozen=True)
class ModelVersion:
    name: str
    model: str
    prompt: str = "v1"

    def __post_init__(self):
        if self.prompt not in PROMPTS:
            raise ValueError(f"unknown prompt variant '{self.prompt}' (known: {', '.join(PROMPTS)})")


def build_messages(version: ModelVersion, user_message: str, doc_ids: list[str], kb: dict) -> list[dict]:
    system = PROMPTS[version.prompt].replace("{context}", context_text(doc_ids, kb))
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message}]


def parse_response(text: str) -> tuple[AssistantResponse | None, str | None]:
    """Return (response, error). Lenient about fences, strict about the schema."""
    try:
        data = parse_json_loose(text)
        if isinstance(data, dict) and isinstance(data.get("tool"), dict) and not data["tool"].get("name"):
            data["tool"] = None  # some models emit {"tool": {}} for "no tool"
        return AssistantResponse.from_dict(data), None
    except (ValueError, SchemaError) as exc:
        return None, str(exc)[:200]
