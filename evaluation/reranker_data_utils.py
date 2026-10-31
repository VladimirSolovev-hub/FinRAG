"""Shared utilities for synthetic reranker data preparation.

Synthetic labels are silver labels. They must never replace the frozen human-reviewed
benchmark used for final test comparisons.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

_CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def document_group_id(doc_id: str) -> str:
    """Group chunks of the same source PDF into one leakage-prevention unit."""
    return _CHUNK_SUFFIX.sub("", clean(doc_id))


def stable_split(group_id: str, validation_percent: int = 10) -> str:
    """Deterministically split document/source groups into train and validation."""
    if not 1 <= validation_percent <= 30:
        raise ValueError("validation_percent должен быть в диапазоне 1..30")
    digest = hashlib.sha256(f"finrag-reranker-split-v1:{group_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "validation" if bucket >= 100 - validation_percent else "train"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSONL {path}, строка {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL {path}, строка {line_no}: ожидался объект")
            rows.append(row)
    return rows


def reserved_groups_from_benchmark(path: Path, splits: set[str] | None = None) -> set[str]:
    """Return source groups used by selected benchmark splits; None means all splits."""
    reserved: set[str] = set()
    if not path.exists():
        return reserved
    for row in load_jsonl(path):
        if splits is not None and clean(row.get("split", "test")) not in splits:
            continue
        ids: list[str] = []
        candidate = clean(row.get("candidate_doc_id"))
        if candidate:
            ids.append(candidate)
        for key in ("gold_doc_ids",):
            values = row.get(key) or []
            if isinstance(values, list):
                ids.extend(clean(item) for item in values)
        relevance = row.get("relevance")
        if isinstance(relevance, dict):
            ids.extend(clean(item) for item in relevance.keys())
        reserved.update(document_group_id(doc_id) for doc_id in ids if doc_id)
    return reserved


def _retry_delay_seconds(headers: Any, detail: str) -> float | None:
    """Read Groq/OpenAI-compatible retry hints from headers or error text."""
    candidates: list[float] = []

    def add_number(value: Any) -> None:
        try:
            number = float(str(value).strip())
            if number >= 0:
                candidates.append(number)
        except (TypeError, ValueError):
            pass

    # Standard Retry-After is usually a number of seconds for these APIs.
    add_number(headers.get("Retry-After") if headers is not None else None)

    # Groq publishes a token-bucket reset hint such as "7.66s" or "1m2.5s".
    reset = headers.get("x-ratelimit-reset-tokens") if headers is not None else None
    if reset:
        duration = str(reset).strip().lower()
        minutes_match = re.search(r"([0-9]+(?:\.[0-9]+)?)m", duration)
        seconds_match = re.search(r"([0-9]+(?:\.[0-9]+)?)s", duration)
        seconds = float(minutes_match.group(1)) * 60 if minutes_match else 0.0
        seconds += float(seconds_match.group(1)) if seconds_match else 0.0
        if seconds > 0:
            candidates.append(seconds)

    # Groq's 429 body often includes: "Please try again in 6.6225s."
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", detail, re.IGNORECASE)
    if match:
        candidates.append(float(match.group(1)))

    return max(candidates) if candidates else None


def chat_json(
    messages: list[dict[str, str]],
    *,
    timeout: int = 120,
    retries: int = 6,
    temperature: float = 0.1,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible Chat Completions endpoint configured via environment."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = (model_override or os.getenv("OPENAI_MODEL", "")).strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    if not api_key or not model:
        raise RuntimeError(
            "Задайте OPENAI_API_KEY и OPENAI_MODEL; при необходимости OPENAI_BASE_URL. "
            "Скрипты поддерживают OpenAI-совместимый Chat Completions API."
        )
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = "unknown error"
    for attempt in range(max(1, retries)):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "finrag-reranker-dataset/1.0",
	        },

            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            outer = json.loads(raw)
            content = outer["choices"][0]["message"]["content"]
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Ответ модели не является JSON-объектом")
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1500]
            last_error = f"HTTP {exc.code}: {detail}"

            # A 429 is not fixed by an arbitrary 1–2 second retry. Respect the
            # provider's explicit wait/reset hints; add a small safety margin.
            if exc.code == 429:
                delay = _retry_delay_seconds(exc.headers, detail)
                if delay is None:
                    delay = min(2 ** (attempt + 1), 60)
                delay = min(delay + 0.5, 120)
                if attempt + 1 < max(1, retries):
                    print(f"[LLM] HTTP 429 rate limit; ждём {delay:.1f} с перед повтором "
                          f"({attempt + 2}/{max(1, retries)} попытка).")
                    time.sleep(delay)
                continue

            # Authentication, permission and malformed-request errors are not
            # transient; don't waste retries on them. Keep retries for 5xx/408/409.
            if exc.code not in {408, 409, 425, 429} and exc.code < 500:
                raise RuntimeError(f"Chat Completions API failed (non-retryable): {last_error}") from exc
        except (urllib.error.URLError, ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = str(exc)

        if attempt + 1 < max(1, retries):
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Chat Completions API failed after retries: {last_error}")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
