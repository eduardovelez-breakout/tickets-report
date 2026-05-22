#!/usr/bin/env python3
"""Export last week's HubSpot tickets to CSV.

Columns:
Created~Ticket Name~Owner~Company~Emails~Category~Subcategory~Summary~Closed~First Reply After
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import sys
import time
import threading
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from typing import Any

DEFAULT_PAT = ""
DEFAULT_API_URL = "https://api.hubapi.com/crm/v3/objects/tickets/search"
DEFAULT_OWNERS_API_URL = "https://api.hubapi.com/crm/v3/owners/"
DEFAULT_OUTPUT = "Tickets - Last 7 Days.csv"
DEFAULT_PROPERTIES = "subject,time_to_first_agent_reply,time_to_close,hubspot_owner_id,content,description,category,support_subcategory,subcategory,hs_ticket_category,hs_ticket_subcategory,hs_ticket_status,hs_pipeline,hs_pipeline_stage"

DEFAULT_GEMINI_API_KEY = ""
DEFAULT_GEMINI_MODEL = "gemma-4-26b-a4b-it"

BLOCKED_COMPANY_NAMES = {"breakout learning", "instructure"}

CONVO_SOURCES: list[tuple[list[str], str, list[str]]] = [
    (["emails", "email"], "emails", ["hs_email_text", "hs_email_html", "hs_body_preview"]),
    (["notes", "note"], "notes", ["hs_note_body", "hs_body_preview"]),
    (["calls", "call"], "calls", ["hs_call_body", "hs_body_preview"]),
    (["tasks", "task"], "tasks", ["hs_task_body", "hs_body_preview"]),
    (["communications", "communication"], "communications", ["hs_communication_body", "hs_body_preview"]),
]


class FixedWindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max(1, int(max_calls))
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._timestamps = deque()

    def acquire(self) -> None:
        while True:
            wait_for = 0.0
            now = time.monotonic()
            with self._lock:
                while self._timestamps and (now - self._timestamps[0]) >= self.window_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_for = self.window_seconds - (now - self._timestamps[0])
            time.sleep(max(0.01, wait_for))


def is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return True
        try:
            body = exc.read().decode("utf-8", errors="ignore").lower()
        except Exception:
            body = ""
        if "rate limit" in body or "resource_exhausted" in body or "quota" in body:
            return True
    msg = str(exc).lower()
    return "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg


def retry_after_seconds(exc: Exception, default_seconds: float) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            value = exc.headers.get("Retry-After")
            if value:
                return max(0.5, float(value))
        except Exception:
            pass
    return max(0.5, float(default_seconds))


def is_server_error(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and 500 <= exc.code <= 599


def isoformat_utc(d: dt.datetime) -> str:
    return d.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        n = int(value)
        if n > 10_000_000_000:
            return dt.datetime.fromtimestamp(n / 1000, tz=dt.timezone.utc)
        return dt.datetime.fromtimestamp(n, tz=dt.timezone.utc)
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        return None

def parse_cli_datetime_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    # Accept YYYY-MM-DD as midnight UTC
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        raw = f"{raw}T00:00:00+00:00"
    elif raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        d = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def format_duration_from_ms(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        n = float(str(value))
    except ValueError:
        return ""
    if n < 100_000:
        n *= 1000
    hours = round(n / 1000 / 3600, 1)
    if hours >= 24:
        days = round(hours / 24, 1)
        return f"{days:g} {'day' if days == 1 else 'days'}"
    return f"{hours:g} {'hr' if hours == 1 else 'hrs'}"


def get_nested(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur




def first_non_empty_paths(obj: dict[str, Any], paths_csv: str) -> str:
    for path in [p.strip() for p in paths_csv.split(",") if p.strip()]:
        v = get_nested(obj, path)
        if v not in (None, ""):
            return str(v)
    return ""


def csv_hyperlink_formula(url: str, label: str) -> str:
    if not url:
        return label
    safe_url = str(url).replace('"', '""')
    safe_label = str(label or url).replace('"', '""')
    return f'=HYPERLINK("{safe_url}","{safe_label}")'




def is_excluded_email(email: str) -> bool:
    e = (email or "").strip().lower()
    if not e:
        return True
    if e.endswith("@breakoutlearning.com"):
        return True
    if "noreply" in e or "no-reply" in e:
        return True
    return False


def extract_emails_from_text(value: str) -> list[str]:
    if not value:
        return []
    found = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", value)
    return [e.lower() for e in found if not is_excluded_email(e)]

def split_emails(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;,\s]+", value)
    return [p.strip().lower() for p in parts if "@" in p and not is_excluded_email(p)]

def fetch_post_json(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_get_json(url: str, token: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def delete_request(url: str, token: str) -> None:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=60):
        return


def fetch_owner_map(owners_api_url: str, token: str, limit: int = 500) -> dict[str, str]:
    owner_map: dict[str, str] = {}
    cursor: str | None = None
    while True:
        params = {"limit": str(limit)}
        if cursor:
            params["after"] = cursor
        payload = fetch_get_json(f"{owners_api_url}?{urllib.parse.urlencode(params)}", token)
        for owner in payload.get("results", []):
            if not isinstance(owner, dict):
                continue
            oid = owner.get("id")
            if oid is None:
                continue
            full_name = f"{str(owner.get('firstName') or '').strip()} {str(owner.get('lastName') or '').strip()}".strip()
            display = full_name or str(owner.get("email") or "").strip() or str(oid)
            owner_map[str(oid)] = display
            if owner.get("userId") is not None:
                owner_map[str(owner.get("userId"))] = display
        cursor = get_nested(payload, "paging.next.after")
        if not cursor:
            break
    return owner_map


def fetch_tickets(
    api_url: str,
    token: str,
    start_iso: str,
    end_iso: str,
    limit: int,
    properties: list[str],
    results_key: str,
    cursor_key: str,
    created_key: str,
    support_pipeline_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    start_dt = parse_ts(start_iso)
    end_dt = parse_ts(end_iso)
    if not start_dt or not end_dt:
        raise RuntimeError("Invalid date range")

    while True:
        body: dict[str, Any] = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "createdate",
                            "operator": "BETWEEN",
                            "value": str(int(start_dt.timestamp() * 1000)),
                            "highValue": str(int(end_dt.timestamp() * 1000)),
                        },
                        {
                            "propertyName": "hs_pipeline",
                            "operator": "EQ",
                            "value": str(support_pipeline_id),
                        },
                    ]
                }
            ],
            "sorts": [{"propertyName": "createdate", "direction": "DESCENDING"}],
            "properties": properties,
            "limit": limit,
        }
        if cursor:
            body["after"] = cursor
        payload = fetch_post_json(api_url, token, body)
        items = payload.get(results_key, [])
        if not isinstance(items, list):
            raise RuntimeError(f"Expected list at key '{results_key}', got: {type(items).__name__}")
        for item in items:
            created = parse_ts(str(get_nested(item, created_key) or ""))
            status_value = first_non_empty_paths(
                item,
                "properties.hs_ticket_status,properties.ticket_status,properties.status",
            ).strip().lower()
            if "spam" in status_value:
                continue
            if created and start_iso <= isoformat_utc(created) <= end_iso:
                out.append(item)
        cursor = get_nested(payload, cursor_key)
        if not cursor:
            break
    return out


def build_ticket_url(ticket: dict[str, Any], ticket_url_key: str, ticket_id_key: str, template: str) -> str:
    explicit = get_nested(ticket, ticket_url_key) if ticket_url_key else None
    if explicit:
        return str(explicit)
    tid = get_nested(ticket, ticket_id_key)
    return template.format(ticket_id=tid) if tid is not None else ""




def strip_known_noise_phrases(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(
        r"messages from Slack, Google Chat, and Microsoft Teams are organized in this note",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def looks_like_prompt_echo(text: str) -> bool:
    t = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not t:
        return False
    bad_markers = [
        "role: internal support-ticket summary writer",
        "constraint 1",
        "constraint 2",
        "sentence 1:",
        "sentence 2:",
        "output exactly 2",
        "hard limit",
    ]
    marker_hits = sum(1 for m in bad_markers if m in t)
    return marker_hits >= 2


def clean_text(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_association_ids(token: str, ticket_id: str, assoc_type: str, limit: int = 100) -> list[str]:
    ids: list[str] = []
    after: str | None = None
    while True:
        params = {"limit": str(limit)}
        if after:
            params["after"] = after
        url = f"https://api.hubapi.com/crm/v3/objects/tickets/{ticket_id}/associations/{assoc_type}?{urllib.parse.urlencode(params)}"
        payload = fetch_get_json(url, token)
        for item in payload.get("results", []):
            if isinstance(item, dict) and item.get("id") is not None:
                ids.append(str(item["id"]))
        after = get_nested(payload, "paging.next.after")
        if not after:
            break
    return ids


def fetch_ticket_contact_ids(token: str, ticket_id: str) -> list[str]:
    for assoc_type in ["contacts", "contact"]:
        try:
            ids = fetch_association_ids(token, ticket_id, assoc_type)
        except Exception:
            ids = []
        if ids:
            return ids
    return []


def fetch_ticket_company_ids(token: str, ticket_id: str) -> list[str]:
    for assoc_type in ["companies", "company"]:
        try:
            ids = fetch_association_ids(token, ticket_id, assoc_type)
        except Exception:
            ids = []
        if ids:
            return ids
    return []


def fetch_objects_batch(token: str, object_type: str, ids: list[str], properties: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    payload = fetch_post_json(
        f"https://api.hubapi.com/crm/v3/objects/{object_type}/batch/read",
        token,
        {"properties": properties, "inputs": [{"id": i} for i in ids]},
    )
    res = payload.get("results", [])
    return res if isinstance(res, list) else []




def contact_is_internal(contact: dict[str, Any]) -> bool:
    props = contact.get("properties", {}) if isinstance(contact, dict) else {}
    if not isinstance(props, dict):
        return False
    raw_values = [str(props.get("email") or ""), str(props.get("hs_additional_emails") or "")]
    emails: set[str] = set()
    for raw in raw_values:
        emails.update(split_emails(raw))
        emails.update(extract_emails_from_text(raw))
    if not emails:
        return False
    return all(is_excluded_email(e) for e in emails)


def sever_internal_ticket_contact_associations(token: str, ticket_id: str, contact_ids: list[str]) -> tuple[list[str], int]:
    if not contact_ids:
        return [], 0
    try:
        contacts = fetch_objects_batch(token, "contacts", contact_ids, ["email", "hs_additional_emails"])
    except Exception:
        return contact_ids, 0

    by_id: dict[str, dict[str, Any]] = {}
    for c in contacts:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if cid:
            by_id[cid] = c

    kept: list[str] = []
    removed_count = 0
    for cid in contact_ids:
        contact = by_id.get(cid)
        if contact and contact_is_internal(contact):
            try:
                delete_request(
                    f"https://api.hubapi.com/crm/v4/objects/tickets/{ticket_id}/associations/contacts/{cid}",
                    token,
                )
            except Exception:
                kept.append(cid)
            else:
                removed_count += 1
            continue
        kept.append(cid)
    return kept, removed_count


def sever_internal_contacts_for_ticket(token: str, ticket_id: str) -> int:
    contact_ids = fetch_ticket_contact_ids(token, ticket_id)
    if not contact_ids:
        return 0
    _, removed = sever_internal_ticket_contact_associations(token, ticket_id, contact_ids)
    return removed


def company_name_is_blocked(name: str) -> bool:
    return normalize_space(name).lower() in BLOCKED_COMPANY_NAMES


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sever_blocked_company_associations_for_ticket(token: str, ticket_id: str) -> int:
    company_ids = fetch_ticket_company_ids(token, ticket_id)
    if not company_ids:
        return 0

    removed = 0
    try:
        companies = fetch_objects_batch(token, "companies", company_ids, ["name"])
    except Exception:
        return 0

    for c in companies:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        props = c.get("properties", {})
        if not isinstance(props, dict):
            continue
        name = normalize_space(str(props.get("name") or ""))
        if company_name_is_blocked(name):
            try:
                delete_request(
                    f"https://api.hubapi.com/crm/v4/objects/tickets/{ticket_id}/associations/companies/{cid}",
                    token,
                )
            except Exception:
                continue
            removed += 1
    return removed


def fetch_ticket_company_name(token: str, ticket_id: str) -> str:
    company_ids = fetch_ticket_company_ids(token, ticket_id)
    if not company_ids:
        return "Unknown"

    try:
        companies = fetch_objects_batch(token, "companies", [company_ids[0]], ["name"])
        if companies and isinstance(companies[0], dict):
            props = companies[0].get("properties", {})
            if isinstance(props, dict):
                name = normalize_space(str(props.get("name") or ""))
                if name and not company_name_is_blocked(name):
                    return name
    except Exception:
        pass

    return "Unknown"


def fetch_ticket_contact_emails(token: str, ticket_id: str) -> str:
    contact_ids = fetch_ticket_contact_ids(token, ticket_id)
    if not contact_ids:
        return ""

    emails: set[str] = set()
    try:
        contacts = fetch_objects_batch(token, "contacts", contact_ids, ["email", "hs_additional_emails"])
        for c in contacts:
            if not isinstance(c, dict):
                continue
            props = c.get("properties", {})
            if not isinstance(props, dict):
                continue
            for raw in [str(props.get("email") or ""), str(props.get("hs_additional_emails") or "")]:
                for e in split_emails(raw):
                    emails.add(e.lower())
    except Exception:
        pass

    return ",".join(sorted(emails))

def fetch_ticket_conversation_text(token: str, ticket: dict[str, Any], ticket_id: str, max_chars: int, debug: bool = False) -> str:
    chunks: list[str] = []
    seen: set[str] = set()

    for assoc_variants, object_type, prop_names in CONVO_SOURCES:
        ids: list[str] = []
        for assoc_type in assoc_variants:
            try:
                ids = fetch_association_ids(token, ticket_id, assoc_type)
            except Exception:
                ids = []
            if ids:
                break

        if not ids:
            continue

        ids = [i for i in ids if not (i in seen or seen.add(i))]
        if not ids:
            continue

        try:
            records = fetch_objects_batch(token, object_type, ids, prop_names)
        except Exception:
            continue

        for rec in records:
            props = rec.get("properties", {}) if isinstance(rec, dict) else {}
            if not isinstance(props, dict):
                continue
            for p in prop_names:
                text = strip_known_noise_phrases(clean_text(props.get(p)))
                if text:
                    chunks.append(text)
                    break

    # Fallback so Summary is never blank when ticket has at least basic text.
    if not chunks:
        for p in ["properties.subject", "properties.content", "properties.description"]:
            text = strip_known_noise_phrases(clean_text(get_nested(ticket, p)))
            if text:
                chunks.append(text)

    joined = "\n\n".join(chunks)[:max_chars]
    if debug:
        print(f"ticket {ticket_id}: convo_chars={len(joined)} chunks={len(chunks)}", file=sys.stderr)
    return joined


def call_gemini_summary(api_key: str, model: str, conversation_text: str, max_chars: int, debug: bool = False, ticket_id: str = "") -> str:
    if not conversation_text:
        return ""
    system_instruction = (
        "You are writing internal support-ticket summaries.\n"
        "Output exactly 2 sentences total (3 only if absolutely needed).\n"
        "Hard limit: 320 characters total.\n"
        "Sentence 1: inferred core issue/root cause, prioritizing support-rep diagnosis over the customer's initial report.\n"
        "Sentence 2: actions/troubleshooting performed by support.\n"
        "Sentence 3 (optional): final status/outcome or resolution.\n"
        "If customer-reported issue conflicts with support findings, prefer support findings.\n"
        "Do not use bullet points, labels, preamble, or extra detail.\n"
        "Do not quote email text. Do not include greetings, signatures, or timestamps. "
        "Use plain factual language and keep it concise."
    )
    user_text = "Conversation:\n" + conversation_text
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": user_text}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.2},
        }
    ).encode("utf-8")
    model_candidates: list[str] = []
    for m in [model, "gemma-4-31b-it", "gemma-3-12b-it"]:
        mm = str(m or "").strip()
        if mm and mm not in model_candidates:
            model_candidates.append(mm)

    payload: dict[str, Any] = {}
    last_exc: Exception | None = None
    for model_name in model_candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model_name)}:generateContent?key={urllib.parse.quote(api_key)}"
        per_model_attempts = 3
        for attempt in range(1, per_model_attempts + 1):
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_exc = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                    # model not available in this project/region; try fallback model
                    break
                if attempt < per_model_attempts and (is_rate_limit_error(exc) or is_server_error(exc)):
                    wait_s = retry_after_seconds(exc, float(attempt))
                    if debug:
                        print(
                            f"ticket {ticket_id}: gemini_model_retry model={model_name} attempt={attempt}/{per_model_attempts} wait={wait_s}",
                            file=sys.stderr,
                        )
                    time.sleep(wait_s)
                    continue
                # non-retriable for this model, move to fallback model (if any)
                break
        if payload:
            break

    if not payload and last_exc:
        raise last_exc

    candidates = payload.get("candidates", [])
    if debug:
        finish = ""
        if isinstance(candidates, list) and candidates:
            finish = str(candidates[0].get("finishReason") or "")
        print(f"ticket {ticket_id}: gemini_candidates={len(candidates) if isinstance(candidates, list) else 0} finish={finish}", file=sys.stderr)

    parts = None
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        content = candidates[0].get("content", {})
        if isinstance(content, dict):
            maybe_parts = content.get("parts")
            if isinstance(maybe_parts, list):
                parts = maybe_parts

    if not isinstance(parts, list):
        if debug:
            print(f"ticket {ticket_id}: gemini_no_parts payload_keys={list(payload.keys())}", file=sys.stderr)
        return ""

    text = "\n".join(
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and not bool(p.get("thought", False))
    ).strip()
    if debug and not text:
        print(f"ticket {ticket_id}: gemini_parts_but_empty_text", file=sys.stderr)
    if looks_like_prompt_echo(text):
        if debug:
            print(f"ticket {ticket_id}: gemini_prompt_echo_rejected", file=sys.stderr)
        return ""
    return text[:max_chars]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export last week's tickets as ~-delimited CSV")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--owners-api-url", default=DEFAULT_OWNERS_API_URL)
    parser.add_argument("--token", help="HubSpot token (overrides DEFAULT_PAT)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-delimiter", default="~", help="CSV delimiter for output (default: ~). Use ',' for standard CSV.")
    parser.add_argument("--gemini-api-key", default=DEFAULT_GEMINI_API_KEY)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)

    parser.add_argument("--results-key", default="results")
    parser.add_argument("--cursor-key", default="paging.next.after")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--properties", default=DEFAULT_PROPERTIES)
    parser.add_argument("--support-pipeline-id", default="0", help="HubSpot hs_pipeline id for Support pipeline")
    parser.add_argument("--start-date", default="", help="UTC/ISO start (inclusive), e.g. 2026-04-04 or 2026-04-04T00:00:00Z")
    parser.add_argument("--end-date", default="", help="UTC/ISO end (exclusive recommended), e.g. 2026-04-11 or 2026-04-11T00:00:00Z")
    parser.add_argument("--conversation-max-chars", type=int, default=8000)
    parser.add_argument("--summary-max-chars", type=int, default=320)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gemini-retries", type=int, default=2)
    parser.add_argument("--gemini-retry-delay", type=float, default=1.0)
    parser.add_argument("--gemini-max-per-minute", type=int, default=10)
    parser.add_argument("--test-20", action="store_true")
    parser.add_argument("--test-20-oldest", action="store_true")
    parser.add_argument("--debug-conversation", action="store_true")
    parser.add_argument("--debug-gemini", action="store_true")
    parser.add_argument("--log-gemini", action="store_true")
    parser.add_argument("--log-gemini-payload-preview", action="store_true")
    parser.add_argument("--log-gemini-payload-chars", type=int, default=500)
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--pending-retries-file", default="report_artifacts/pending_summary_retries.csv")
    parser.add_argument("--retry-pending-only", action="store_true")

    parser.add_argument("--created-key", default="createdAt")
    parser.add_argument("--closed-key", default="properties.time_to_close")
    parser.add_argument("--first-reply-key", default="properties.time_to_first_agent_reply")
    parser.add_argument("--owner-key", default="properties.hubspot_owner_id")
    parser.add_argument("--ticket-name-key", default="properties.subject,properties.name")
    parser.add_argument("--category-key", default="properties.hs_ticket_category,properties.category")
    parser.add_argument("--subcategory-key", default="properties.support_subcategory,properties.hs_ticket_subcategory,properties.subcategory")
    parser.add_argument("--ticket-url-key", default="")
    parser.add_argument("--ticket-id-key", default="id")
    parser.add_argument("--ticket-url-template", default="https://app.hubspot.com/help-desk/39763539/view/100528752/ticket/{ticket_id}")

    args = parser.parse_args()
    token = args.token or DEFAULT_PAT
    if not token:
        print("Missing HubSpot token", file=sys.stderr)
        return 1
    if not args.gemini_api_key:
        print("Missing Gemini API key", file=sys.stderr)
        return 1

    gemini_limiter = FixedWindowRateLimiter(args.gemini_max_per_minute, 60.0)

    now = dt.datetime.now(dt.timezone.utc)
    start_dt = parse_cli_datetime_utc(args.start_date) if args.start_date else (now - dt.timedelta(days=7))
    end_dt = parse_cli_datetime_utc(args.end_date) if args.end_date else now
    if not start_dt or not end_dt:
        print("Invalid --start-date/--end-date. Use YYYY-MM-DD or ISO datetime.", file=sys.stderr)
        return 1
    if end_dt <= start_dt:
        print("--end-date must be after --start-date", file=sys.stderr)
        return 1
    start_iso = isoformat_utc(start_dt)
    end_iso = isoformat_utc(end_dt)
    properties = [p.strip() for p in args.properties.split(",") if p.strip()]

    try:
        tickets = fetch_tickets(
            args.api_url,
            token,
            start_iso,
            end_iso,
            args.limit,
            properties,
            args.results_key,
            args.cursor_key,
            args.created_key,
            args.support_pipeline_id,
        )
        if args.test_20_oldest:
            tickets = tickets[-20:]
        elif args.test_20:
            tickets = tickets[:20]
        owner_map = fetch_owner_map(args.owners_api_url, token)
    except Exception as exc:
        print(f"Failed to fetch HubSpot data: {exc}", file=sys.stderr)
        return 1

    existing_pending_rows: list[dict[str, str]] = []
    pending_by_id: dict[str, dict[str, str]] = {}
    if os.path.exists(args.pending_retries_file):
        try:
            with open(args.pending_retries_file, "r", encoding="utf-8", newline="") as pf:
                for row in csv.DictReader(pf):
                    tid = str(row.get("ticket_id") or "").strip()
                    if tid:
                        existing_pending_rows.append(row)
                        pending_by_id[tid] = row
        except Exception:
            existing_pending_rows = []
            pending_by_id = {}

    if args.retry_pending_only:
        pending_ids = {str(r.get("ticket_id") or "").strip() for r in existing_pending_rows if str(r.get("ticket_id") or "").strip()}
        tickets = [t for t in tickets if str(get_nested(t, args.ticket_id_key) or "").strip() in pending_ids]
        if not tickets:
            print(f"No pending tickets matched date range in {args.pending_retries_file}")
            return 0

    removed_contact_total = 0
    removed_company_total = 0
    ticket_ids_for_sever = [str(get_nested(t, args.ticket_id_key) or "") for t in tickets]
    for tid in ticket_ids_for_sever:
        if not tid:
            continue
        try:
            removed_contact_total += sever_internal_contacts_for_ticket(token, tid)
            removed_company_total += sever_blocked_company_associations_for_ticket(token, tid)
        except Exception:
            continue

    if removed_contact_total > 0 or removed_company_total > 0:
        print(
            f"Severed {removed_contact_total} internal contact and {removed_company_total} blocked company associations; waiting 60s for HubSpot association enrichment",
            file=sys.stderr,
        )
        time.sleep(60)

    def process_ticket(t: dict[str, Any]) -> tuple[dict[str, str], dict[str, str] | None]:
        created = parse_ts(str(get_nested(t, args.created_key) or ""))
        ticket_id = str(get_nested(t, args.ticket_id_key) or "")
        convo = fetch_ticket_conversation_text(token, t, ticket_id, args.conversation_max_chars, args.debug_conversation) if ticket_id else ""
        if args.log_gemini_payload_preview and convo:
            preview_len = max(50, int(args.log_gemini_payload_chars))
            preview = re.sub(r"\s+", " ", convo[:preview_len]).strip()
            print(f"ticket {ticket_id}: gemini_payload_preview chars={len(convo)} preview={preview}", file=sys.stderr)
        summary = ""
        gemini_failed = False
        gemini_failure_reason = ""
        if convo and not args.skip_gemini:
            attempts = max(0, args.gemini_retries) + 1
            for attempt in range(1, attempts + 1):
                if args.log_gemini:
                    print(f"ticket {ticket_id}: gemini_start attempt={attempt}/{attempts}", file=sys.stderr)
                try:
                    gemini_limiter.acquire()
                    summary = call_gemini_summary(
                        args.gemini_api_key,
                        args.gemini_model,
                        convo,
                        args.summary_max_chars,
                        debug=args.debug_gemini,
                        ticket_id=ticket_id,
                    )
                    if args.log_gemini:
                        print(f"ticket {ticket_id}: gemini_done attempt={attempt}/{attempts} chars={len(summary)}", file=sys.stderr)
                except Exception as exc:
                    if args.debug_conversation or args.log_gemini:
                        print(f"ticket {ticket_id}: gemini_error attempt={attempt}/{attempts} err={exc}", file=sys.stderr)
                    summary = ""
                    gemini_failure_reason = str(exc)

                    if attempt < attempts and is_rate_limit_error(exc):
                        wait_s = retry_after_seconds(exc, max(args.gemini_retry_delay * attempt, 1.0))
                        if args.log_gemini:
                            print(f"ticket {ticket_id}: gemini_rate_limit_backoff seconds={wait_s}", file=sys.stderr)
                        time.sleep(wait_s)
                        continue

                if summary:
                    break

                if attempt < attempts:
                    wait_s = max(0.0, args.gemini_retry_delay) * attempt
                    if args.log_gemini:
                        print(f"ticket {ticket_id}: gemini_retry_wait seconds={wait_s}", file=sys.stderr)
                    time.sleep(wait_s)

        if convo and not summary:
            if args.debug_conversation or args.log_gemini:
                print(f"ticket {ticket_id}: gemini_empty_fallback", file=sys.stderr)
            if not args.skip_gemini:
                gemini_failed = True
                if not gemini_failure_reason:
                    gemini_failure_reason = "empty_response"
            summary = convo[:args.summary_max_chars]

        summary = re.sub(r"\s+", " ", str(summary or "")).strip()
        if args.skip_gemini:
            summary = summary[:20]

        owner_id = str(get_nested(t, args.owner_key) or "")
        ticket_url = build_ticket_url(t, args.ticket_url_key, args.ticket_id_key, args.ticket_url_template)
        ticket_name = first_non_empty_paths(t, args.ticket_name_key) or ticket_url
        company_name = fetch_ticket_company_name(token, ticket_id) if ticket_id else ""
        emails = fetch_ticket_contact_emails(token, ticket_id) if ticket_id else ""
        if not emails:
            email_set = set(extract_emails_from_text(convo))
            email_set.update(extract_emails_from_text(str(first_non_empty_paths(t, "properties.content,properties.description"))))
            emails = ",".join(sorted(email_set))
        category = first_non_empty_paths(t, args.category_key)
        subcategory = first_non_empty_paths(t, args.subcategory_key)
        row = {
            "Created": created.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "") if created else "",
            "Ticket Name": csv_hyperlink_formula(ticket_url, ticket_name),
            "Owner": owner_map.get(owner_id, owner_id),
            "Company": company_name,
            "Emails": emails,
            "Category": category,
            "Subcategory": subcategory,
            "Summary": summary,
            "Closed": format_duration_from_ms(get_nested(t, args.closed_key)),
            "First Reply After": format_duration_from_ms(get_nested(t, args.first_reply_key)),
        }
        pending_row: dict[str, str] | None = None
        if gemini_failed and ticket_id:
            pending_row = {
                "ticket_id": ticket_id,
                "ticket_url": ticket_url,
                "created": row["Created"],
                "owner": row["Owner"],
                "company": row["Company"],
                "category": row["Category"],
                "subcategory": row["Subcategory"],
                "failure_reason": gemini_failure_reason[:400],
                "conversation_excerpt": re.sub(r"\s+", " ", convo[:2000]).strip(),
            }
        return row, pending_row

    rows: list[dict[str, str]] = []
    pending_failures_by_id: dict[str, dict[str, str]] = {}
    attempted_ticket_ids: set[str] = set()
    worker_count = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(process_ticket, t) for t in tickets]
        future_to_ticket_id = {fut: str(get_nested(t, args.ticket_id_key) or "").strip() for fut, t in zip(futures, tickets)}
        for fut in as_completed(futures):
            tid = future_to_ticket_id.get(fut, "")
            if tid:
                attempted_ticket_ids.add(tid)
            try:
                row, pending_row = fut.result()
                rows.append(row)
                if pending_row and pending_row.get("ticket_id"):
                    pending_failures_by_id[pending_row["ticket_id"]] = pending_row
            except Exception as exc:
                if args.debug_conversation:
                    print(f"ticket_worker_error={exc}", file=sys.stderr)

    rows.sort(key=lambda r: r["Created"], reverse=True)
    delim = str(args.csv_delimiter or "~")
    if delim == "\\t":
        delim = "\t"
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Created", "Ticket Name", "Owner", "Company", "Emails", "Category", "Subcategory", "Summary", "Closed", "First Reply After"],
            delimiter=delim,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    final_pending_by_id: dict[str, dict[str, str]] = {}
    for tid, row in pending_by_id.items():
        if tid not in attempted_ticket_ids:
            final_pending_by_id[tid] = row
    final_pending_by_id.update(pending_failures_by_id)

    pending_fields = [
        "ticket_id",
        "ticket_url",
        "created",
        "owner",
        "company",
        "category",
        "subcategory",
        "failure_reason",
        "conversation_excerpt",
    ]
    os.makedirs(os.path.dirname(args.pending_retries_file) or ".", exist_ok=True)
    with open(args.pending_retries_file, "w", newline="", encoding="utf-8") as pf:
        pw = csv.DictWriter(pf, fieldnames=pending_fields)
        pw.writeheader()
        for tid in sorted(final_pending_by_id.keys()):
            src = final_pending_by_id[tid]
            pw.writerow({k: str(src.get(k) or "") for k in pending_fields})

    print(f"Wrote {len(rows)} tickets to {args.output}")
    print(f"Pending summary retries: {len(final_pending_by_id)} -> {args.pending_retries_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
