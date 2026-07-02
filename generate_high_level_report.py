#!/usr/bin/env python3
"""Generate a weekly trend report from ticket CSV with deterministic institution counts.

Outputs:
- report_artifacts/enriched_tickets.csv
- report_artifacts/company_counts.json
- report_artifacts/trend_insights.json
- report_artifacts/final_report.md
- report_artifacts/final_report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_CSV = "Tickets - Last 7 Days.csv"
DEFAULT_OUTDIR = "report_artifacts"


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


def _is_rate_limited(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return True
        try:
            body = exc.read().decode("utf-8", errors="ignore").lower()
        except Exception:
            body = ""
        return "rate" in body or "quota" in body or "resource_exhausted" in body
    msg = str(exc).lower()
    return "rate" in msg or "quota" in msg or "resource_exhausted" in msg


def _retry_after_seconds(exc: Exception, fallback_seconds: float) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            v = exc.headers.get("Retry-After")
            if v:
                return max(0.5, float(v))
        except Exception:
            pass
    return max(0.5, float(fallback_seconds))


def _is_model_not_found(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and exc.code == 404


def read_rows(csv_path: Path, delimiter: str) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(r) for r in reader]


def write_rows(csv_path: Path, rows: list[dict[str, str]], delimiter: str) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str, limit: int = 500) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s[:limit]


def normalize_text(v: str) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def normalize_person_name(value: str) -> str:
    name = normalize_text(value)
    if not name or re.fullmatch(r"\d{6,}", name):
        return ""
    return name




def derive_date_range_label(rows: list[dict[str, str]]) -> str:
    dates: list[str] = []
    for r in rows:
        created = normalize_text(r.get("Created", ""))
        if not created:
            continue
        date_part = created.split("T", 1)[0]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_part):
            dates.append(date_part)
    if not dates:
        return ""
    return f"{min(dates)} to {max(dates)}"


def derive_display_date_range(rows: list[dict[str, str]]) -> str:
    raw = derive_date_range_label(rows)
    if not raw or " to " not in raw:
        return raw
    start_s, end_s = raw.split(" to ", 1)
    try:
        start = datetime_from_ymd(start_s)
        end = datetime_from_ymd(end_s)
    except ValueError:
        return raw
    if start.year == end.year:
        return f"{start.strftime('%b')} {start.day} - {end.strftime('%b')} {end.day}, {end.year}"
    return f"{start.strftime('%b')} {start.day}, {start.year} - {end.strftime('%b')} {end.day}, {end.year}"


def datetime_from_ymd(value: str) -> Any:
    from datetime import datetime

    return datetime.strptime(value, "%Y-%m-%d")


def build_ticket_corpus(rows: list[dict[str, str]], max_rows: int) -> str:
    lines: list[str] = []
    for i, r in enumerate(rows[:max_rows], start=1):
        line = (
            f"[{i}] "
            f"Created={r.get('Created','')} | "
            f"Owner={r.get('Owner','')} | "
            f"Company={r.get('Company','')} | "
            f"Category={r.get('Category','')} | "
            f"Subcategory={r.get('Subcategory','')} | "
            f"Summary={clean_text(r.get('Summary',''), 350)}"
        )
        lines.append(line)
    return "\n".join(lines)


def call_gemini(api_key: str, model: str, prompt: str, limiter: FixedWindowRateLimiter) -> str:
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are an internal support analytics assistant. "
                        "Return only the requested format. "
                        "Do not repeat instructions."
                    )
                }
            ]
        },
        "generationConfig": {"temperature": 0.2},
    }

    model_candidates: list[str] = []
    for m in [model, "gemma-4-31b-it", "gemma-3-12b-it"]:
        mm = str(m or "").strip()
        if mm and mm not in model_candidates:
            model_candidates.append(mm)

    last_exc: Exception | None = None
    payload: dict[str, Any] = {}
    for model_name in model_candidates:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model_name)}:generateContent?key={urllib.parse.quote(api_key)}"
        )
        attempts = 4
        for attempt in range(1, attempts + 1):
            limiter.acquire()
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_exc = exc
                if _is_model_not_found(exc):
                    # Try the next fallback model immediately.
                    break
                if attempt >= attempts:
                    # Exhausted retries for this model; move to next fallback model.
                    break
                if _is_rate_limited(exc):
                    time.sleep(_retry_after_seconds(exc, 3.0 * attempt))
                else:
                    time.sleep(1.0 * attempt)
        if payload:
            break

    if not payload and last_exc:
        raise last_exc

    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
    parts = content.get("parts", []) if isinstance(content, dict) else []
    if not isinstance(parts, list):
        return ""
    return "\n".join(
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and not bool(p.get("thought", False))
    ).strip()


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("No JSON found in model output")


def infer_missing_categories(
    rows: list[dict[str, str]],
    api_key: str,
    model: str,
    limiter: FixedWindowRateLimiter,
) -> list[dict[str, str]]:
    missing = []
    for i, r in enumerate(rows, start=1):
        cat = normalize_text(r.get("Category", ""))
        sub = normalize_text(r.get("Subcategory", ""))
        if not cat or not sub:
            missing.append((i, r))

    if not missing:
        return rows

    known_categories = sorted({normalize_text(r.get("Category", "")) for r in rows if normalize_text(r.get("Category", ""))})
    known_subcategories = sorted({normalize_text(r.get("Subcategory", "")) for r in rows if normalize_text(r.get("Subcategory", ""))})

    ticket_lines = []
    for idx, r in missing[:200]:
        ticket_lines.append(
            f"[{idx}] Category={normalize_text(r.get('Category','')) or '<missing>'} | "
            f"Subcategory={normalize_text(r.get('Subcategory','')) or '<missing>'} | "
            f"Summary={clean_text(r.get('Summary',''), 700)}"
        )

    prompt = f"""
Backfill missing support ticket Category/Subcategory values.

Return STRICT JSON object:
{{
  "updates": [
    {{"index": 1, "category": "", "subcategory": "", "confidence": "high|medium|low"}}
  ]
}}

Rules:
- Only include rows where at least one value is missing.
- Reuse existing taxonomy when possible.
- If unsure, set category="Other" and subcategory="Other".
- Keep labels concise and consistent.

Known categories:
{known_categories}

Known subcategories:
{known_subcategories}

Tickets needing backfill:
{chr(10).join(ticket_lines)}
""".strip()

    text = call_gemini(api_key, model, prompt, limiter)
    data = extract_json(text)
    updates = data.get("updates", []) if isinstance(data, dict) else []

    for u in updates:
        if not isinstance(u, dict):
            continue
        idx = int(u.get("index", 0))
        if idx < 1 or idx > len(rows):
            continue
        cat = normalize_text(u.get("category", ""))
        sub = normalize_text(u.get("subcategory", ""))
        if cat:
            rows[idx - 1]["Category"] = cat
        if sub:
            rows[idx - 1]["Subcategory"] = sub

    return rows


def compute_company_counts(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    counts = Counter(normalize_text(r.get("Company", "")) or "Unknown" for r in rows)
    account_manager_counts: dict[str, Counter[str]] = {}
    for r in rows:
        company = normalize_text(r.get("Company", "")) or "Unknown"
        account_manager = (
            normalize_person_name(r.get("Account Manager", ""))
            or normalize_person_name(r.get("Company Owner", ""))
            or normalize_person_name(r.get("Owner", ""))
            or "Unknown"
        )
        account_manager_counts.setdefault(company, Counter())[account_manager] += 1
    unknown_count = counts.get("Unknown", 0)
    ranked_counts = Counter({k: v for k, v in counts.items() if k != "Unknown"})
    total_ranked = sum(ranked_counts.values())
    out = []
    for company, cnt in ranked_counts.most_common():
        account_manager = "Unknown"
        if company in account_manager_counts and account_manager_counts[company]:
            account_manager = account_manager_counts[company].most_common(1)[0][0]
        out.append({
            "company": company,
            "account_manager": account_manager,
            "ticket_count": cnt,
            "share_pct": round((cnt / total_ranked) * 100, 1) if total_ranked else 0.0,
        })
    return out, unknown_count


def compute_class_code_company_counts(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    class_re = re.compile(r"\b([A-Z]{2,6}\s?-?\d{2,4}[A-Z]?)\b")
    counts: Counter[tuple[str, str]] = Counter()

    for r in rows:
        company = normalize_text(r.get("Company", "")) or "Unknown"
        if company == "Unknown":
            continue
        blob = " ".join([
            normalize_text(r.get("Ticket Name", "")),
            normalize_text(r.get("Summary", "")),
            normalize_text(r.get("Category", "")),
            normalize_text(r.get("Subcategory", "")),
        ])
        codes = {m.group(1).strip().upper().replace("  ", " ") for m in class_re.finditer(blob)}
        for code in codes:
            counts[(code, company)] += 1

    out: list[dict[str, Any]] = []
    for (code, company), n in counts.most_common(limit):
        out.append({"code": code, "company": company, "ticket_count": n})
    return out


def render_class_code_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None detected"]
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        company = str(item.get("company", "")).strip()
        n = int(item.get("ticket_count", 0) or 0)
        if code and company:
            lines.append(f"- {code} ({company}) - {n} tickets")
    return lines or ["- None detected"]


def extract_ticket_url(row: dict[str, str]) -> str:
    raw = str(row.get("Ticket Name", "") or "").strip()
    if not raw:
        return ""
    m = re.search(r'=HYPERLINK\("([^"]+)"', raw)
    if m:
        return m.group(1)
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    m = re.search(r"https://app\.hubspot\.com/help-desk/[^\s\"]+", raw)
    if m:
        return m.group(0)
    return ""


def format_citation_labels(indexes: Any, rows: list[dict[str, str]]) -> str:
    if not isinstance(indexes, list):
        return ""
    seen: list[int] = []
    for v in indexes:
        try:
            n = int(v)
        except Exception:
            continue
        if n > 0 and n <= len(rows) and n not in seen:
            seen.append(n)
    if not seen:
        return ""

    links: list[str] = []
    for n in seen:
        url = extract_ticket_url(rows[n - 1])
        if url:
            links.append(f"[{n}]({url})")
        else:
            links.append(str(n))
    return " (" + ", ".join(links) + ")"

def evidence_links_from_indexes(indexes: Any, rows: list[dict[str, str]]) -> list[str]:
    links: list[str] = []
    if not isinstance(indexes, list):
        return links
    for v in indexes:
        try:
            i = int(v)
        except Exception:
            continue
        if i < 1 or i > len(rows):
            continue
        url = extract_ticket_url(rows[i - 1])
        if url:
            links.append(url)
    seen: set[str] = set()
    out: list[str] = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out



def render_trend_text_sections(trends_json: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []

    lines.append("## Key Trends")
    key_trends = trends_json.get("key_trends", []) if isinstance(trends_json, dict) else []
    if isinstance(key_trends, list) and key_trends:
        for t in key_trends:
            if not isinstance(t, dict):
                continue
            trend = str(t.get("trend", "")).strip() or "Unlabeled trend"
            why = str(t.get("why_it_matters", "")).strip()
            cite = format_citation_labels(t.get("evidence_ticket_indexes"), rows)
            if len(trend.split()) <= 3 and why:
                lines.append(f"- **{trend}**: {why}{cite}")
            else:
                lines.append(f"- **{trend}**: {why}{cite}" if why else f"- **{trend}**{cite}")
    else:
        lines.append("- No clear trend output returned.")
    return lines


def strip_markdown_links(value: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    return re.sub(r"\s+", " ", s).strip()


def html_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", normalize_text(name)) if p]
    if not parts or normalize_text(name).lower() == "unknown":
        return "?"
    return "".join(p[0].upper() for p in parts[:2])


def ticket_word(count: int) -> str:
    return "ticket" if count == 1 else "tickets"


def ticket_refs(indexes: list[int]) -> str:
    if not indexes:
        return ""
    if len(indexes) == 1:
        return f"Ticket #{indexes[0]}"
    shown = ", ".join(f"#{i}" for i in indexes[:5])
    if len(indexes) > 5:
        shown += f", +{len(indexes) - 5}"
    return f"Tickets {shown}"


def top_issue_tags(rows: list[dict[str, str]], max_tags: int = 2) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in ("Subcategory", "Category"):
            value = normalize_text(row.get(key, ""))
            if "@" in value or value.startswith("http"):
                continue
            if value and value.lower() not in {"unknown", "other", "none"}:
                counts[value] += 1
    return [tag for tag, _ in counts.most_common(max_tags)]


def company_row_indexes(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, row in enumerate(rows, start=1):
        company = normalize_text(row.get("Company", "")) or "Unknown"
        out.setdefault(company, []).append(i)
    return out


def render_html_trends(trends_json: dict[str, Any]) -> str:
    trends = trends_json.get("key_trends", []) if isinstance(trends_json, dict) else []
    items: list[str] = []
    for trend in trends[:3] if isinstance(trends, list) else []:
        if not isinstance(trend, dict):
            continue
        label = strip_markdown_links(str(trend.get("trend", "")).strip())
        why = strip_markdown_links(str(trend.get("why_it_matters", "")).strip())
        text = label if not why else f"{label} - {why}"
        if text:
            items.append(f"<p class=\"trend-line\"><span class=\"bullet\">&bull;&nbsp;</span>{html_escape(text)}</p>")
    if not items:
        items.append("<p class=\"trend-line\"><span class=\"bullet\">&bull;&nbsp;</span>No clear systemic trends identified this week.</p>")
    return "\n".join(items)


def render_html_trend_box(trends_json: dict[str, Any]) -> str:
    return (
        '<table class="trend-box"><tr><td>'
        '<p class="trend-title">TRENDS THIS WEEK</p>'
        f"{render_html_trends(trends_json)}"
        "</td></tr></table>"
    )


def render_html_tag_table(count: int, tags: list[str]) -> str:
    cells = [f'<td class="tag-cell count-tag"><b>{count} {ticket_word(count)}</b></td>']
    cells.extend(f'<td class="tag-cell issue-tag"><b>{html_escape(tag)}</b></td>' for tag in tags[:2])
    return f'<table class="tag-table"><tr>{"".join(cells)}</tr></table>'


def render_html_account_sections(
    rows: list[dict[str, str]],
    company_counts: list[dict[str, Any]],
    institution_insight_map: dict[str, dict[str, str]],
    limit: int = 20,
) -> str:
    indexes_by_company = company_row_indexes(rows)
    rows_by_company: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        company = normalize_text(row.get("Company", "")) or "Unknown"
        rows_by_company.setdefault(company, []).append(row)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for company_row in company_counts[:limit]:
        am = normalize_text(str(company_row.get("account_manager") or "")) or "Unassigned accounts"
        if am.lower() == "unknown":
            am = "Unassigned accounts"
        grouped.setdefault(am, []).append(company_row)

    sections: list[str] = []
    for account_manager in sorted(grouped.keys()):
        companies = grouped[account_manager]
        total = sum(int(c.get("ticket_count", 0) or 0) for c in companies)
        sections.append(
            "<section class=\"account-section\">"
            "<table class=\"am-header\"><tr>"
            f"<td class=\"am-initials\">{html_escape(initials(account_manager))}</td>"
            f"<td class=\"am-name\">{html_escape(account_manager)}</td>"
            f"<td class=\"am-count\"><b>{total} {ticket_word(total)}</b></td>"
            "</tr></table>"
        )
        for company_row in companies:
            company = str(company_row.get("company", "")).strip()
            count = int(company_row.get("ticket_count", 0) or 0)
            source_indexes = indexes_by_company.get(company, [])
            source_rows = rows_by_company.get(company, [])
            insight = institution_insight_map.get(company.lower(), {})
            trend = strip_markdown_links(insight.get("summary", ""))
            if not trend:
                trend = (
                    f"{count} {ticket_word(count)} this week. No systemic pattern identified."
                    if count == 1
                    else f"{count} {ticket_word(count)} this week. Review the related tickets for repeated account-specific friction."
                )
            tags = top_issue_tags(source_rows)
            tag_html = render_html_tag_table(count, tags)
            action = strip_markdown_links(insight.get("next_step", ""))
            if account_manager == "Unassigned accounts":
                action = "Assign account ownership so someone can follow up."
            elif not action and (count > 1 or tags):
                action = "Review whether the account needs proactive follow-up."
            action_html = (
                f'<table class="action-table"><tr><td><p class="action">&rarr; {html_escape(action)}</p></td></tr></table>'
                if action
                else '<p class="action empty">&nbsp;</p>'
            )
            sections.append(
                '<table class="institution-card"><tr><td class="institution-card-cell">'
                "<table class=\"institution-head\"><tr>"
                f"<td class=\"institution-name\">{html_escape(company)}</td>"
                f"<td class=\"ticket-refs\">{html_escape(ticket_refs(source_indexes) or f'{count} {ticket_word(count)}')}</td>"
                "</tr></table>"
                f"<p class=\"institution-summary\">{html_escape(trend)}</p>"
                f"{tag_html}"
                f"{action_html}"
                "</td></tr></table>"
            )
        sections.append("</section>")
    return "\n".join(sections)


def render_final_report_html(
    rows: list[dict[str, str]],
    company_counts: list[dict[str, Any]],
    trends_json: dict[str, Any],
    institution_insight_map: dict[str, dict[str, str]],
) -> str:
    total_tickets = len(rows)
    institution_count = len(company_counts)
    key_trends = trends_json.get("key_trends", []) if isinstance(trends_json, dict) else []
    trend_count = len(key_trends) if isinstance(key_trends, list) else 0
    display_range = derive_display_date_range(rows)
    range_line = f"Weekly support report &middot; {html_escape(display_range)}" if display_range else "Weekly support report"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin:0; padding:54pt; max-width:504pt; color:#111827; font-family:Arial, sans-serif; background:#ffffff; }}
    p {{ margin:0; }}
    .eyebrow {{ color:#6b7280; font-size:9pt; margin-bottom:4pt; }}
    .title {{ color:#1a3a5c; font-size:24pt; font-weight:700; margin-bottom:2pt; }}
    .title span {{ font-size:20pt; }}
    .subtitle {{ color:#6b7280; font-size:10pt; margin-bottom:16pt; }}
    .stats {{ border-collapse:collapse; margin:12pt 0 18pt 0; width:450pt; }}
    .stats td {{ border:1pt solid #d1d5db; background:#f3f4f6; padding:6pt 8pt; width:150pt; vertical-align:top; }}
    .stat-label {{ color:#6b7280; font-size:8pt; }}
    .stat-value {{ color:#1a3a5c; font-size:24pt; font-weight:700; margin-top:3pt; }}
    .trend-box {{ border-collapse:collapse; margin:0 0 22pt 0; width:468pt; }}
    .trend-box td {{ border:1pt solid #f59e0b; border-left:2.2pt solid #f59e0b; background:#fef3c7; padding:7pt 9pt; vertical-align:top; }}
    .trend-title {{ color:#92400e; font-size:9pt; font-weight:700; margin-bottom:6pt; }}
    .trend-line {{ color:#374151; font-size:10pt; line-height:1.25; margin:3pt 0; }}
    .bullet {{ color:#92400e; padding-right:6pt; }}
    .section-label {{ color:#6b7280; font-size:9pt; font-weight:700; margin:12pt 0 8pt 0; }}
    .account-section {{ margin:0 0 18pt 0; padding-top:10pt; }}
    .am-header {{ border-collapse:collapse; width:468pt; margin-bottom:8pt; }}
    .am-initials {{ width:36pt; background:#d6e8f7; color:#2e6da4; font-size:10pt; font-weight:700; text-align:center; padding:4pt; border-top:0.8pt solid #d1d5db; border-left:0pt solid #ffffff; border-right:0pt solid #ffffff; border-bottom:0pt solid #ffffff; }}
    .am-name {{ width:342pt; color:#1a3a5c; font-size:13pt; font-weight:700; padding:4pt 8pt; border-top:0.8pt solid #d1d5db; border-left:0pt solid #ffffff; border-right:0pt solid #ffffff; border-bottom:0pt solid #ffffff; }}
    .am-count {{ width:90pt; color:#6b7280; font-size:10pt; font-weight:700; text-align:right; padding:4pt; border-top:0.8pt solid #d1d5db; border-left:0pt solid #ffffff; border-right:0pt solid #ffffff; border-bottom:0pt solid #ffffff; }}
    .institution-card {{ border-collapse:collapse; width:468pt; margin:0 0 8pt 0; page-break-inside:avoid; break-inside:avoid; }}
    .institution-card-cell {{ border:1pt solid #d1d5db; background:#ffffff; padding:8pt; vertical-align:top; }}
    .institution-head {{ border-collapse:collapse; width:100%; }}
    .institution-name {{ color:#1a3a5c; font-size:11pt; font-weight:700; width:70%; border:0pt solid #ffffff; }}
    .ticket-refs {{ color:#6b7280; font-size:9pt; text-align:right; width:30%; border:0pt solid #ffffff; }}
    .institution-summary {{ color:#374151; font-size:10pt; line-height:1.3; margin:8pt 0; }}
    .tag-table {{ border-collapse:collapse; margin:4pt 0 7pt 0; }}
    .tag-cell {{ font-size:8pt; font-weight:700; padding:3pt 8pt; border:0; }}
    .count-tag {{ background:#f3f4f6; color:#6b7280; }}
    .issue-tag {{ background:#fef3c7; color:#92400e; }}
    .action-table {{ border-collapse:collapse; width:100%; margin-top:4pt; }}
    .action-table td {{ border-top:0.5pt solid #d1d5db; border-left:0pt solid #ffffff; border-right:0pt solid #ffffff; border-bottom:0pt solid #ffffff; padding-top:5pt; }}
    .action {{ color:#2e6da4; font-size:9pt; }}
    .action.empty {{ color:#ffffff; }}
    .footer {{ background:#f3f4f6; color:#6b7280; font-size:9pt; padding:7pt 8pt; margin-top:18pt; width:450pt; }}
  </style>
</head>
<body>
  <p class="eyebrow">{range_line}</p>
  <p class="title">Support <span>at a glance</span></p>
  <p class="subtitle">Review issues, spot patterns, and reach out proactively where needed.</p>
  <table class="stats"><tr>
    <td><p class="stat-label">TOTAL TICKETS</p><p class="stat-value">{total_tickets}</p></td>
    <td><p class="stat-label">INSTITUTIONS AFFECTED</p><p class="stat-value">{institution_count}</p></td>
    <td><p class="stat-label">SYSTEMIC TRENDS</p><p class="stat-value">{trend_count}</p></td>
  </tr></table>
  {render_html_trend_box(trends_json)}
  <p class="section-label">ACCOUNTS</p>
  {render_html_account_sections(rows, company_counts, institution_insight_map, limit=20)}
  <p class="footer">Questions about a specific ticket? Reach out to the support team. This report is generated weekly.</p>
</body>
</html>
"""


def build_institution_insight_map(trends_json: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    patterns = trends_json.get("institution_friction_patterns", []) if isinstance(trends_json, dict) else []
    if not isinstance(patterns, list):
        return out
    for p in patterns:
        if not isinstance(p, dict):
            continue
        inst = str(p.get("institution", "")).strip()
        if not inst:
            continue
        arr = p.get("patterns", [])
        patterns_text = ", ".join(str(x).strip() for x in arr if str(x).strip()) if isinstance(arr, list) else ""
        cite = format_citation_labels(p.get("evidence_ticket_indexes"), rows)
        next_step = str(p.get("next_step", "")).strip()
        out[inst.lower()] = {
            "summary": (patterns_text + cite).strip() or ("No specific pattern provided" + cite),
            "next_step": next_step,
        }
    return out


def build_institution_trend_map(trends_json: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, str]:
    insights = build_institution_insight_map(trends_json, rows)
    return {institution: data.get("summary", "") for institution, data in insights.items()}


def build_company_markdown_tables_by_account_manager(
    company_counts: list[dict[str, Any]],
    institution_trend_map: dict[str, str],
    limit: int = 15,
) -> str:
    account_managers: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, row in enumerate(company_counts[:limit], start=1):
        account_manager = str(row.get("account_manager") or "Unknown").strip() or "Unknown"
        account_managers.setdefault(account_manager, []).append((i, row))

    lines: list[str] = []
    for account_manager in sorted(account_managers.keys()):
        lines.extend([
            f"### {account_manager}",
            "",
            "| Rank | Institution | Tickets | Share | Institutional Trends |",
            "|---:|---|---:|---:|---|",
        ])
        for rank, row in account_managers[account_manager]:
            company = str(row.get("company", ""))
            trend_text = institution_trend_map.get(company.lower(), "")
            lines.append(
                f"| {rank} | {company} | {row['ticket_count']} | {row['share_pct']}% | {trend_text} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_fallback_trends_json(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Deterministic fallback when Gemini is unavailable."""
    category_counts = Counter(normalize_text(r.get("Category", "")) or "Other" for r in rows)
    subcategory_counts = Counter(normalize_text(r.get("Subcategory", "")) or "Other" for r in rows)

    key_trends: list[dict[str, Any]] = []
    for category, count in category_counts.most_common(5):
        if category.lower() == "unknown":
            continue
        top_sub = ""
        top_sub_count = 0
        for sub, sub_count in subcategory_counts.items():
            if sub_count > top_sub_count:
                top_sub = sub
                top_sub_count = sub_count
        trend_label = f"Several tickets cluster around {category.lower()} workflows"
        why = f"{count} tickets mention this area in the current reporting period."
        if top_sub and top_sub.lower() != "other":
            why = f"{count} tickets, often tied to {top_sub.lower()}."
        key_trends.append(
            {
                "trend": trend_label[:120],
                "why_it_matters": why[:140],
                "evidence_ticket_indexes": [],
            }
        )

    return {
        "key_trends": key_trends or [
            {
                "trend": "No model-generated trend output available",
                "why_it_matters": "Gemini call failed; report generated from deterministic counts only.",
                "evidence_ticket_indexes": [],
            }
        ],
        "institution_friction_patterns": [],
        "data_quality_caveats": [
            "Gemini trend generation unavailable in this run; fallback trends are deterministic."
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly trend report with deterministic institution counts")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--gemini-max-per-minute", type=int, default=20)
    parser.add_argument("--csv-delimiter", default="~", help="Input/output CSV delimiter (default: ~). Use ',' for standard CSV.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing Gemini API key. Pass --api-key or set GEMINI_API_KEY")

    delim = str(args.csv_delimiter or "~")
    if delim == "\\t":
        delim = "\t"

    rows = read_rows(Path(args.csv), delimiter=delim)
    if not rows:
        raise SystemExit("CSV has no rows")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gemini_limiter = FixedWindowRateLimiter(args.gemini_max_per_minute, 60.0)
    try:
        rows = infer_missing_categories(rows, args.api_key, args.model, gemini_limiter)
    except Exception as exc:
        print(f"Warning: category backfill skipped due to Gemini error: {exc}")

    enriched_csv = outdir / "enriched_tickets.csv"
    write_rows(enriched_csv, rows, delimiter=delim)

    company_counts, unknown_company_count = compute_company_counts(rows)
    (outdir / "company_counts.json").write_text(json.dumps(company_counts, indent=2), encoding="utf-8")

    corpus = build_ticket_corpus(rows, args.max_rows)
    prompt_trends = f"""
You are analyzing support-ticket trends for universities/institutions.

Return STRICT JSON with this exact shape:
{{
  "key_trends": [
    {{"trend": "", "why_it_matters": "", "evidence_ticket_indexes": [1,2,3]}}
  ],
  "institution_friction_patterns": [
    {{"institution": "", "patterns": ["", ""], "next_step": "", "evidence_ticket_indexes": [1,2]}}
  ],
  "data_quality_caveats": [""]
}}

Rules:
- Focus on trend surfacing plus one short account-manager follow-up step per institution.
- `next_step` should be an account-manager-facing next step, at most 16 words.
- Keep `next_step` specific and practical; do not create broad action plans.
- Leave `next_step` blank when there is no useful follow-up.
- Do not invent institutions; use provided data.
- Keep outputs concise and factual.
- Keep every `trend` to at most 16 words.
- Keep every `why_it_matters` to at most 20 words.
- Keep each institution pattern item to at most 10 words.
- Prefer short, direct phrasing over full narrative.
- Determine core issue from support-rep diagnosis/resolution and final outcome signals, not from the user's initial claim alone.
- If initial report conflicts with support findings, prefer support findings as source of truth.
- Treat reported student issues as unverified symptoms, not confirmed bugs.
- User error, misunderstanding, and configuration/workflow mismatch are common; prefer those interpretations unless clear evidence indicates a product defect.
- Do not label something a bug/defect/outage unless evidence strongly supports it.
- Trend names must be specific behavioral patterns with context (channel/workflow/platform), not broad buckets.
- Bad trend labels (do not use): "Login Issues", "Technical Issues", "Group Problems", "Billing Questions".
- Good trend label style: "Students authenticated through SSO instead of Canvas LTI and could not enter assigned breakout groups".
- Prefer formulations like: "Several users/students/instructors ..." and include the concrete trigger/failure mode.

Deterministic institution counts:
{json.dumps(company_counts[:25])}

Ticket rows:
{corpus}
""".strip()

    try:
        trends_text = call_gemini(args.api_key, args.model, prompt_trends, gemini_limiter)
        trends_json = extract_json(trends_text)
    except Exception as exc:
        print(f"Warning: trend generation failed; using deterministic fallback: {exc}")
        trends_json = build_fallback_trends_json(rows)
    (outdir / "trend_insights.json").write_text(json.dumps(trends_json, indent=2), encoding="utf-8")
    institution_insight_map = build_institution_insight_map(
        trends_json if isinstance(trends_json, dict) else {},
        rows,
    )
    institution_trend_map = {k: v.get("summary", "") for k, v in institution_insight_map.items()}

    date_range_label = derive_date_range_label(rows)
    title = "# Weekly Ticket Trend Report"
    if date_range_label:
        title = f"# Weekly Ticket Trend Report ({date_range_label})"

    md_parts = [
        title,
        "",
    ]
    md_parts.extend(render_trend_text_sections(trends_json if isinstance(trends_json, dict) else {}, rows))
    md_parts.extend([
        "",
        "## Top Institutions By Ticket Volume",
        build_company_markdown_tables_by_account_manager(company_counts, institution_trend_map, limit=20),
        "",
    ])
    (outdir / "final_report.md").write_text("\n".join(md_parts), encoding="utf-8")
    (outdir / "final_report.html").write_text(
        render_final_report_html(
            rows,
            company_counts,
            trends_json if isinstance(trends_json, dict) else {},
            institution_insight_map,
        ),
        encoding="utf-8",
    )

    print(f"Wrote report artifacts to: {outdir}")
    print(f"Enriched CSV: {enriched_csv}")
    print(f"Company counts: {outdir / 'company_counts.json'}")
    print(f"Trend insights: {outdir / 'trend_insights.json'}")
    print(f"Final report: {outdir / 'final_report.md'}")
    print(f"Final HTML report: {outdir / 'final_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
