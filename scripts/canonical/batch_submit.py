"""OpenAI Batch API helpers for the Stage 2 corrector.

The Stage 2 pipeline has two LLM calls per row (corrector → reviewer),
with the reviewer depending on corrector output. To run sequentially
via Batch API:

  1. submit corrector.jsonl  (this module)
  2. poll until done          (this module)
  3. download + parse output  (this module)
  4. run sync reviewer on non-noop rows  (scripts/stage2_batch.py)

Each JSONL line is a `/v1/chat/completions` request with a unique
`custom_id` of the form "<shabad_id>_<batch_idx>". The callsite keeps a
`custom_id → [clip_ids]` mapping to re-link responses back to rows.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from .corrector import (
    CORRECTOR_SYSTEM_PROMPT, CorrectorConfig, build_batch_corrector_prompt,
)


def build_corrector_jsonl(
    candidates: list[dict],
    shabad_lines_map: dict[str, list[str]],
    output_path: Path,
    cfg: CorrectorConfig,
) -> dict[str, list[str]]:
    """Write the OpenAI Batch API JSONL file. Returns a
    {custom_id: [clip_ids]} mapping so the resolver can rebuild
    per-row results from the batched responses.

    Rows with no retrieved shabad are skipped (they shouldn't be
    Stage 2 candidates, but defensive)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        sid = c.get("shabad_id") or ""
        if sid and sid in shabad_lines_map:
            groups[sid].append(c)

    custom_id_to_clips: dict[str, list[str]] = {}
    with open(output_path, "w", encoding="utf-8") as f:
        for sid, rows in groups.items():
            lines = shabad_lines_map[sid]
            for bi, chunk_start in enumerate(range(0, len(rows), cfg.batch_size)):
                chunk = rows[chunk_start:chunk_start + cfg.batch_size]
                custom_id = f"{sid}_{bi}"
                items = [
                    {
                        "clip_id": r["clip_id"],
                        "caption": (
                            r.get("text_cleaned")
                            or r.get("text")
                            or ""
                        ),
                        "sttm_hint": (
                            r.get("final_text")
                            or r.get("canonical_text")
                        ),
                    }
                    for r in chunk
                ]
                prompt = build_batch_corrector_prompt(items, lines)
                body = {
                    "model": cfg.model,
                    "messages": [
                        {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": cfg.max_output_tokens,
                    "response_format": {"type": "json_object"},
                }
                if cfg.temperature is not None:
                    body["temperature"] = cfg.temperature
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
                custom_id_to_clips[custom_id] = [r["clip_id"] for r in chunk]
    return custom_id_to_clips


def _openai_client(client=None):
    if client is not None:
        return client
    from openai import OpenAI
    key = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("missing OPEN_API_KEY / OPENAI_API_KEY")
    return OpenAI(api_key=key)


def submit_batch(
    jsonl_path: Path,
    completion_window: str = "24h",
    metadata: dict | None = None,
    client=None,
) -> dict:
    """Upload JSONL + create batch. Returns
    {batch_id, input_file_id, status}."""
    client = _openai_client(client)
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata=metadata or {},
    )
    return {
        "batch_id": batch.id,
        "input_file_id": upload.id,
        "status": batch.status,
    }


def poll_batch(batch_id: str, client=None) -> dict:
    client = _openai_client(client)
    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
        } if counts else None,
    }


def download_batch_output(
    output_file_id: str, dest_path: Path, client=None,
) -> None:
    client = _openai_client(client)
    resp = client.files.content(output_file_id)
    content = resp.read() if hasattr(resp, "read") else resp.content
    if isinstance(content, str):
        content = content.encode("utf-8")
    Path(dest_path).write_bytes(content)


def parse_corrector_responses(
    output_jsonl: Path,
) -> tuple[dict[str, str], list[str]]:
    """Parse OpenAI Batch API output JSONL → ({clip_id: corrected},
    [error_custom_ids]).

    Missing clip_ids are absent from the returned dict — callers should
    fall back to caption verbatim and label the row `error_fallback`."""
    results: dict[str, str] = {}
    errors: list[str] = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cid_for_batch = entry.get("custom_id")
            if entry.get("error"):
                errors.append(cid_for_batch)
                continue
            body = (entry.get("response") or {}).get("body") or {}
            choice = (body.get("choices") or [{}])[0]
            raw = (choice.get("message") or {}).get("content") or ""
            if not raw:
                errors.append(cid_for_batch)
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                errors.append(cid_for_batch)
                continue
            for c in data.get("corrections", []):
                cid = c.get("clip_id")
                corrected = c.get("corrected")
                if cid and corrected:
                    results[cid] = corrected
    return results, errors


# ---------------------------------------------------------------------------
# Gemini Batch API (reviewer)
# ---------------------------------------------------------------------------

def build_reviewer_jsonl_gemini(
    needs_review: dict[str, list[dict]],
    shabad_lines_map: dict[str, list[str]],
    output_path: Path,
    cfg,  # ReviewerConfig
) -> dict[str, list[str]]:
    """Write Gemini Batch API input JSONL (one request per line). Each
    request is a batched reviewer prompt covering up to cfg.batch_size
    items. Returns a {custom_key: [clip_ids]} mapping."""
    # Local import to avoid circular import (reviewer.py imports this module
    # only indirectly).
    from .reviewer import REVIEWER_SYSTEM_PROMPT, build_batch_reviewer_prompt

    custom_key_to_clips: dict[str, list[str]] = {}
    with open(output_path, "w", encoding="utf-8") as f:
        for sid, items in needs_review.items():
            lines = shabad_lines_map.get(sid, [])
            if not lines:
                continue
            for bi, start in enumerate(range(0, len(items), cfg.batch_size)):
                chunk = items[start:start + cfg.batch_size]
                custom_key = f"{sid}_r{bi}"
                prompt = build_batch_reviewer_prompt(chunk, lines)
                full_prompt = REVIEWER_SYSTEM_PROMPT + "\n\n" + prompt
                max_tokens = cfg.per_item_max_output_tokens * len(chunk) + 256
                request_body = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": full_prompt}],
                    }],
                    "generation_config": {
                        "temperature": cfg.temperature,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                    },
                }
                entry = {"key": custom_key, "request": request_body}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                custom_key_to_clips[custom_key] = [it["clip_id"] for it in chunk]
    return custom_key_to_clips


def _gemini_client(client=None):
    if client is not None:
        return client
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("missing GEMINI_API_KEY")
    return genai.Client(api_key=key)


def submit_gemini_batch(
    jsonl_path: Path,
    model: str,
    display_name: str | None = None,
    client=None,
) -> dict:
    """Upload JSONL + create Gemini batch job. Returns
    {batch_name, input_file_name, state}."""
    client = _gemini_client(client)
    uploaded = client.files.upload(
        file=str(jsonl_path),
        config={"display_name": display_name or "reviewer-batch-input",
                "mime_type": "jsonl"},
    )
    batch = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": display_name or "reviewer-batch"},
    )
    return {
        "batch_name": batch.name,
        "input_file_name": uploaded.name,
        "state": batch.state.name if hasattr(batch.state, "name") else str(batch.state),
    }


def poll_gemini_batch(batch_name: str, client=None) -> dict:
    """Return {state, dest_file_name?, request_counts}."""
    client = _gemini_client(client)
    batch = client.batches.get(name=batch_name)
    state = batch.state.name if hasattr(batch.state, "name") else str(batch.state)
    dest = getattr(batch, "dest", None)
    dest_file_name = getattr(dest, "file_name", None) if dest else None
    counts = {}
    if hasattr(batch, "request_counts") and batch.request_counts:
        for key in ("total", "succeeded", "failed"):
            counts[key] = getattr(batch.request_counts, key, None)
    return {
        "state": state,
        "dest_file_name": dest_file_name,
        "request_counts": counts,
    }


def download_gemini_batch_output(
    file_name: str, dest_path: Path, client=None,
) -> None:
    """Download Gemini batch output file."""
    client = _gemini_client(client)
    content = client.files.download(file=file_name)
    if isinstance(content, str):
        content = content.encode("utf-8")
    Path(dest_path).write_bytes(content)


def parse_reviewer_responses_gemini(
    output_jsonl: Path,
) -> tuple[dict[str, dict], list[str]]:
    """Parse Gemini Batch output JSONL → ({clip_id: review_dict},
    [error_custom_keys]).

    Each review_dict has {verdict, final_text, changed_tokens, reasons}."""
    results: dict[str, dict] = {}
    errors: list[str] = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = entry.get("key") or entry.get("custom_key")
            if entry.get("error") or entry.get("status") == "FAILED":
                errors.append(key)
                continue
            resp = entry.get("response") or {}
            # Gemini returns the model response directly. Look for the text.
            candidates = resp.get("candidates") or []
            if not candidates:
                errors.append(key)
                continue
            parts = (candidates[0].get("content") or {}).get("parts") or []
            raw = ""
            for p in parts:
                if "text" in p:
                    raw += p["text"]
            if not raw:
                errors.append(key)
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                errors.append(key)
                continue
            for review in data.get("reviews", []):
                cid = review.get("clip_id")
                if not cid:
                    continue
                verdict = (review.get("verdict") or "REJECT").upper()
                if verdict not in ("APPROVE", "IMPROVED", "REJECT"):
                    verdict = "REJECT"
                results[cid] = {
                    "verdict": verdict,
                    "final_text": review.get("final_text"),
                    "changed_tokens": review.get("changed_tokens") or [],
                    "reasons": review.get("reasons") or [],
                    "reasoning": review.get("reasoning"),
                }
    return results, errors
