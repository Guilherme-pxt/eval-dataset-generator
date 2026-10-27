"""Small text utilities shared by every step.

Everything here is deterministic and dependency-free so that the same
helpers can be used for clustering, failure signals, deterministic scoring
and the oracle reviewer without the definitions drifting apart.
"""

from __future__ import annotations

import json
import re

STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at for from by with about as into over
    is are was were be been being am do does did done have has had having
    i me my we our you your it its this that these those there here what which
    who whom how when where why can could would should will shall may might must
    not no so than too very just also any some all each per
    hi hey hello thanks thank please pls quick question ok okay guys team folks
    there's what's i'm i've it's don't doesn't can't
    much many long get need like want way go know
    """.split()
)

_NUM_RE = re.compile(r"(?<![\w.])\$?(\d+(?:[.,]\d+)*)")
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_WORD_RE = re.compile(r"[a-z][a-z0-9']*")
_INVOICE_RE = re.compile(r"\bINV-\d{4,}\b", re.IGNORECASE)
_ACCOUNT_RE = re.compile(r"\bACC-\d{4,}\b", re.IGNORECASE)

REFUSAL_PATTERNS = [
    r"\bi can(?:'|no)t help\b",
    r"\bi cannot help\b",
    r"\bi(?:'m| am) (?:not able|unable) to\b",
    r"\bi can(?:'|no)t assist\b",
    r"\bi cannot assist\b",
    r"\bi(?:'m| am) sorry,? but i can\b",
    r"\bcan(?:'|no)t provide (?:that|this|information)\b",
    r"\bnot able to (?:help|assist|provide)\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def normalize_number(raw: str) -> str:
    """Canonical form of a number token: no thousands separators, no trailing zeros."""
    s = raw.replace(",", "")
    if s.count(".") > 1:
        s = s.replace(".", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    s = s.lstrip("0") or "0"
    if s.startswith("."):
        s = "0" + s
    return s


def extract_numbers(text: str, ignore_list_markers: bool = True) -> set[str]:
    """Return the set of normalized numbers mentioned in ``text``.

    Line-leading list markers ("1. ", "2) ") are ignored by default because a
    numbered list is formatting, not a factual claim.
    """
    if ignore_list_markers:
        text = _LIST_MARKER_RE.sub(" ", text)
    # IDs such as INV-20931 are identifiers, not quantities.
    text = _INVOICE_RE.sub(" ", text)
    text = _ACCOUNT_RE.sub(" ", text)
    out = set()
    for m in _NUM_RE.finditer(text):
        token = m.group(1).rstrip(".,")
        if token:
            out.add(normalize_number(token))
    return out


def stem(word: str) -> str:
    """Very small plural stripper; good enough for matching 'days' with 'day'."""
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def content_words(text: str) -> list[str]:
    text = text.lower()
    text = _INVOICE_RE.sub(" invid ", text)
    text = _ACCOUNT_RE.sub(" accid ", text)
    text = re.sub(r"\d+(?:[.,]\d+)*", " ", text)
    words = _WORD_RE.findall(text)
    return [stem(w.strip("'")) for w in words if w not in STOPWORDS and len(w) > 1]


def tokenize_for_clustering(text: str) -> list[str]:
    """Tokens used by TF-IDF: content words with IDs and numbers abstracted."""
    t = text.lower()
    t = _INVOICE_RE.sub(" invoiceid ", t)
    t = _ACCOUNT_RE.sub(" accountid ", t)
    t = re.sub(r"\d+(?:[.,]\d+)*", " ", t)
    words = _WORD_RE.findall(t)
    return [stem(w.strip("'")) for w in words if w not in STOPWORDS and len(w) > 1]


def fact_present(fact: str, text: str) -> bool:
    """Is a short required fact (e.g. "$29 per month") stated in ``text``?

    Rule, chosen so a reviewer can predict every verdict:
    * a fact with numbers is present when all of its numbers appear in the text;
      the words around a number are paraphrased too often to be evidence
      ("2.9% + $0.30 per transaction" vs "the fee is 2.9% + $0.30");
    * a fact without numbers ("CSV or PDF", "Scale plan") needs 60% of its
      content words.
    Wrong numbers are caught separately by the unsupported-number check.
    """
    fact_nums = extract_numbers(fact, ignore_list_markers=False)
    if fact_nums:
        return fact_nums <= extract_numbers(text, ignore_list_markers=False)
    fact_words = set(content_words(fact))
    if not fact_words:
        return fact.strip().lower() in text.lower()
    coverage = len(fact_words & set(content_words(text))) / len(fact_words)
    return coverage >= 0.6


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def find_invoice_ids(text: str) -> list[str]:
    return [m.group(0).upper() for m in _INVOICE_RE.finditer(text or "")]


def find_account_ids(text: str) -> list[str]:
    return [m.group(0).upper() for m in _ACCOUNT_RE.finditer(text or "")]


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


def redact_pii(text: str) -> tuple[str, int]:
    """Replace e-mail addresses and phone numbers. Returns (text, n_redactions)."""
    n = 0

    def _sub(token):
        def inner(_m):
            nonlocal n
            n += 1
            return token
        return inner

    text = _EMAIL_RE.sub(_sub("[EMAIL]"), text)
    text = _PHONE_RE.sub(_sub("[PHONE]"), text)
    return text, n


def parse_json_loose(text: str):
    """Parse a JSON object from model output that may include fences or prose."""
    if text is None:
        raise ValueError("empty model output")
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError(f"no JSON object found in model output: {text[:120]!r}")
