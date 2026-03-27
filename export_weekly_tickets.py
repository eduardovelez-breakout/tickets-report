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
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from typing import Any

DEFAULT_PAT = ""
DEFAULT_API_URL = "https://api.hubapi.com/crm/v3/objects/tickets/search"
DEFAULT_OWNERS_API_URL = "https://api.hubapi.com/crm/v3/owners/"
DEFAULT_OUTPUT = "Tickets - Last 7 Days.csv"
DEFAULT_PROPERTIES = "subject,time_to_first_agent_reply,time_to_close,hubspot_owner_id,content,description,category,support_subcategory,subcategory,hs_ticket_category,hs_ticket_subcategory"

DEFAULT_GEMINI_API_KEY = ""
DEFAULT_GEMINI_MODEL = "gemma-3-27b-it"

CONVO_SOURCES: list[tuple[list[str], str, list[str]]] = [
    (["emails", "email"], "emails", ["hs_email_text", "hs_email_html", "hs_body_preview"]),
    (["notes", "note"], "notes", ["hs_note_body", "hs_body_preview"]),
    (["calls", "call"], "calls", ["hs_call_body", "hs_body_preview"]),
    (["tasks", "task"], "tasks", ["hs_task_body", "hs_body_preview"]),
    (["communications", "communication"], "communications", ["hs_communication_body", "hs_body_preview"]),
]


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


def fetch_tickets(api_url: str, token: str, start_iso: str, end_iso: str, limit: int, properties: list[str], results_key: str, cursor_key: str, created_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    start_dt = parse_ts(start_iso)
    end_dt = parse_ts(end_iso)
    if not start_dt or not end_dt:
        raise RuntimeError("Invalid date range")

    while True:
        body: dict[str, Any] = {
            "filterGroups": [{"filters": [{"propertyName": "createdate", "operator": "BETWEEN", "value": str(int(start_dt.timestamp() * 1000)), "highValue": str(int(end_dt.timestamp() * 1000))}]}],
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




def fetch_ticket_company_name(token: str, ticket_id: str) -> str:
    company_ids: list[str] = []
    for assoc_type in ["companies", "company"]:
        try:
            company_ids = fetch_association_ids(token, ticket_id, assoc_type)
        except Exception:
            company_ids = []
        if company_ids:
            break
    if not company_ids:
        return ""

    try:
        companies = fetch_objects_batch(token, "companies", [company_ids[0]], ["name"])
        if companies and isinstance(companies[0], dict):
            props = companies[0].get("properties", {})
            if isinstance(props, dict):
                name = str(props.get("name") or "").strip()
                if name:
                    return name
    except Exception:
        pass

    return f"ID {company_ids[0]}"


def fetch_ticket_contact_emails(token: str, ticket_id: str) -> str:
    contact_ids: list[str] = []
    for assoc_type in ["contacts", "contact"]:
        try:
            contact_ids = fetch_association_ids(token, ticket_id, assoc_type)
        except Exception:
            contact_ids = []
        if contact_ids:
            break
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
                text = clean_text(props.get(p))
                if text:
                    chunks.append(text)
                    break

    # Fallback so Summary is never blank when ticket has at least basic text.
    if not chunks:
        for p in ["properties.subject", "properties.content", "properties.description"]:
            text = clean_text(get_nested(ticket, p))
            if text:
                chunks.append(text)

    joined = "\n\n".join(chunks)[:max_chars]
    if debug:
        print(f"ticket {ticket_id}: convo_chars={len(joined)} chunks={len(chunks)}", file=sys.stderr)
    return joined


def call_gemini_summary(api_key: str, model: str, conversation_text: str, max_chars: int, debug: bool = False, ticket_id: str = "") -> str:
    if not conversation_text:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(api_key)}"
    prompt = (
        "You are writing internal support-ticket summaries.\n"
        "Output exactly 2 or 3 sentences total.\n"
        "Sentence 1: customer issue/context.\n"
        "Sentence 2: actions/troubleshooting performed.\n"
        "Sentence 3 (optional): current status/outcome.\n"
        "Do not quote email text. Do not include greetings, signatures, or timestamps. "
        "Use plain factual language and keep it concise.\n\n"
        "Conversation:\n" + conversation_text
    )
    req = urllib.request.Request(
        url,
        data=json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

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

    text = "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
    if debug and not text:
        print(f"ticket {ticket_id}: gemini_parts_but_empty_text", file=sys.stderr)
    return text[:max_chars]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export last week's tickets as ~-delimited CSV")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--owners-api-url", default=DEFAULT_OWNERS_API_URL)
    parser.add_argument("--token", help="HubSpot token (overrides DEFAULT_PAT)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--gemini-api-key", default=DEFAULT_GEMINI_API_KEY)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)

    parser.add_argument("--results-key", default="results")
    parser.add_argument("--cursor-key", default="paging.next.after")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--properties", default=DEFAULT_PROPERTIES)
    parser.add_argument("--conversation-max-chars", type=int, default=8000)
    parser.add_argument("--summary-max-chars", type=int, default=900)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gemini-retries", type=int, default=2)
    parser.add_argument("--gemini-retry-delay", type=float, default=1.0)
    parser.add_argument("--test-20", action="store_true")
    parser.add_argument("--test-20-oldest", action="store_true")
    parser.add_argument("--debug-conversation", action="store_true")
    parser.add_argument("--debug-gemini", action="store_true")
    parser.add_argument("--log-gemini", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")

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

    now = dt.datetime.now(dt.timezone.utc)
    start_iso = isoformat_utc(now - dt.timedelta(days=7))
    end_iso = isoformat_utc(now)
    properties = [p.strip() for p in args.properties.split(",") if p.strip()]

    try:
        tickets = fetch_tickets(args.api_url, token, start_iso, end_iso, args.limit, properties, args.results_key, args.cursor_key, args.created_key)
        if args.test_20_oldest:
            tickets = tickets[-20:]
        elif args.test_20:
            tickets = tickets[:20]
        owner_map = fetch_owner_map(args.owners_api_url, token)
    except Exception as exc:
        print(f"Failed to fetch HubSpot data: {exc}", file=sys.stderr)
        return 1

    def process_ticket(t: dict[str, Any]) -> dict[str, str]:
        created = parse_ts(str(get_nested(t, args.created_key) or ""))
        ticket_id = str(get_nested(t, args.ticket_id_key) or "")
        convo = fetch_ticket_conversation_text(token, t, ticket_id, args.conversation_max_chars, args.debug_conversation) if ticket_id else ""
        summary = ""
        if convo and not args.skip_gemini:
            attempts = max(0, args.gemini_retries) + 1
            for attempt in range(1, attempts + 1):
                if args.log_gemini:
                    print(f"ticket {ticket_id}: gemini_start attempt={attempt}/{attempts}", file=sys.stderr)
                try:
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

                if summary:
                    break

                if attempt < attempts:
                    if args.log_gemini:
                        print(f"ticket {ticket_id}: gemini_retry_wait seconds={max(0.0, args.gemini_retry_delay) * attempt}", file=sys.stderr)
                    time.sleep(max(0.0, args.gemini_retry_delay) * attempt)

        if convo and not summary:
            if args.debug_conversation or args.log_gemini:
                print(f"ticket {ticket_id}: gemini_empty_fallback", file=sys.stderr)
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
        return {
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

    rows: list[dict[str, str]] = []
    worker_count = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(process_ticket, t) for t in tickets]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as exc:
                if args.debug_conversation:
                    print(f"ticket_worker_error={exc}", file=sys.stderr)

    rows.sort(key=lambda r: r["Created"], reverse=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Created", "Ticket Name", "Owner", "Company", "Emails", "Category", "Subcategory", "Summary", "Closed", "First Reply After"], delimiter="~", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tickets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
