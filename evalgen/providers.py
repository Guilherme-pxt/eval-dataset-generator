"""LLM providers behind one small interface.

``Provider.chat(model, messages, json_schema=None) -> ChatResult`` is all the
pipeline needs. Implementations:

* ``OllamaProvider``           local server, POST /api/chat (used for every real run)
* ``OpenAICompatibleProvider`` any /v1/chat/completions endpoint; configured from
                               env vars, implemented but not exercised against a paid API
* ``ScriptedProvider``         deterministic fake for tests
* ``CachedProvider``           sqlite3 response cache wrapping any of the above

Decoding is pinned (temperature, seed) so reruns are comparable; the cache
makes them exactly repeatable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import requests


@dataclass
class ChatResult:
    text: str
    model: str
    latency_s: float
    cached: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Provider(Protocol):
    name: str

    def chat(self, model: str, messages: list[dict], json_schema: dict | None = None) -> ChatResult: ...

    def release(self, model: str) -> None:
        """Free resources held for ``model`` (e.g. unload it from memory)."""


class ProviderError(RuntimeError):
    pass


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", temperature: float = 0.0, seed: int = 42,
                 timeout: float = 300.0, num_ctx: int | None = None, keep_alive: str = "5m"):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def options(self) -> dict:
        opts = {"temperature": self.temperature, "seed": self.seed}
        if self.num_ctx:
            opts["num_ctx"] = self.num_ctx
        return opts

    def payload(self, model: str, messages: list[dict], json_schema: dict | None) -> dict:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # reasoning models: answer directly, keeps latency bounded
            "keep_alive": self.keep_alive,
            "options": self.options(),
        }
        if json_schema is not None:
            body["format"] = json_schema
        return body

    def chat(self, model: str, messages: list[dict], json_schema: dict | None = None) -> ChatResult:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=self.payload(model, messages, json_schema),
                              timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"ollama request failed: {exc}") from exc
        if r.status_code != 200:
            raise ProviderError(f"ollama returned HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        return ChatResult(
            text=data.get("message", {}).get("content", ""),
            model=model,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def release(self, model: str) -> None:
        """Unload the model so the next one has the memory (16 GB machine)."""
        try:
            requests.post(f"{self.base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        except requests.RequestException:
            pass

    def cache_identity(self) -> dict:
        return {"provider": self.name, "options": self.options(), "think": False}


class OpenAICompatibleProvider:
    """Chat Completions over plain HTTP. Reads EVALGEN_OPENAI_BASE_URL and EVALGEN_OPENAI_API_KEY."""

    name = "openai_compatible"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 seed: int = 42, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("EVALGEN_OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("EVALGEN_OPENAI_API_KEY")
        if not self.base_url:
            raise ProviderError("set EVALGEN_OPENAI_BASE_URL (e.g. https://host/v1)")
        if not self.api_key:
            raise ProviderError("set EVALGEN_OPENAI_API_KEY")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout

    def payload(self, model: str, messages: list[dict], json_schema: dict | None) -> dict:
        body = {"model": model, "messages": messages, "temperature": self.temperature, "seed": self.seed}
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        return body

    def chat(self, model: str, messages: list[dict], json_schema: dict | None = None) -> ChatResult:
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=self.payload(model, messages, json_schema),
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"request failed: {exc}") from exc
        if r.status_code != 200:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        usage = data.get("usage") or {}
        return ChatResult(
            text=data["choices"][0]["message"]["content"] or "",
            model=model,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def release(self, model: str) -> None:
        return None

    def cache_identity(self) -> dict:
        return {"provider": self.name, "base_url": self.base_url, "temperature": self.temperature, "seed": self.seed}


class ScriptedProvider:
    """Test double: ``fn(model, messages, json_schema) -> str``."""

    name = "scripted"

    def __init__(self, fn: Callable[[str, list[dict], dict | None], str]):
        self.fn = fn
        self.calls: list[dict] = []
        self.released: list[str] = []

    def chat(self, model: str, messages: list[dict], json_schema: dict | None = None) -> ChatResult:
        self.calls.append({"model": model, "messages": messages, "json_schema": json_schema})
        return ChatResult(text=self.fn(model, messages, json_schema), model=model, latency_s=0.0)

    def release(self, model: str) -> None:
        self.released.append(model)

    def cache_identity(self) -> dict:
        return {"provider": self.name}


class CachedProvider:
    """sqlite3-backed response cache. Key = hash of provider identity, model, messages, schema."""

    def __init__(self, inner, path: str | Path):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT, text TEXT, "
            "latency_s REAL, prompt_tokens INTEGER, completion_tokens INTEGER, created_at TEXT)"
        )
        self.db.commit()
        self.hits = 0
        self.misses = 0

    def _key(self, model, messages, json_schema) -> str:
        blob = json.dumps(
            {"id": self.inner.cache_identity(), "model": model, "messages": messages, "schema": json_schema},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, model: str, messages: list[dict], json_schema: dict | None = None) -> ChatResult:
        key = self._key(model, messages, json_schema)
        with self._lock:
            row = self.db.execute(
                "SELECT text, latency_s, prompt_tokens, completion_tokens FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row:
            self.hits += 1
            return ChatResult(text=row[0], model=model, latency_s=row[1], cached=True,
                              prompt_tokens=row[2], completion_tokens=row[3])
        self.misses += 1
        res = self.inner.chat(model, messages, json_schema)
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                (key, model, res.text, res.latency_s, res.prompt_tokens, res.completion_tokens),
            )
            self.db.commit()
        return res

    def release(self, model: str) -> None:
        self.inner.release(model)

    def cache_identity(self) -> dict:
        return self.inner.cache_identity()

    def close(self) -> None:
        self.db.close()


def build_provider(cfg: dict, cache_path: str | Path | None = None):
    """Create the provider described by the [llm] config table."""
    kind = cfg.get("provider", "ollama")
    if kind == "ollama":
        inner = OllamaProvider(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=float(cfg.get("temperature", 0.0)),
            seed=int(cfg.get("seed", 42)),
            timeout=float(cfg.get("timeout_s", 300)),
            num_ctx=cfg.get("num_ctx"),
        )
    elif kind == "openai_compatible":
        inner = OpenAICompatibleProvider(temperature=float(cfg.get("temperature", 0.0)), seed=int(cfg.get("seed", 42)))
    else:
        raise ValueError(f"unknown provider '{kind}'")
    return CachedProvider(inner, cache_path) if cache_path else inner
