#!/usr/bin/env python3
"""Generate a weekly trend report from ticket CSV with deterministic institution counts.

Outputs:
- report_artifacts/enriched_tickets.csv
- report_artifacts/company_counts.json
- report_artifacts/trend_insights.json
- report_artifacts/final_report.md
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_MODEL = "gemma-4-31b-it"
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


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="~")
        return [dict(r) for r in reader]


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="~")
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str, limit: int = 500) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s[:limit]


def normalize_text(v: str) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()




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
    for m in [model, "gemma-3-12b-it"]:
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
    return "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()


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
    unknown_count = counts.get("Unknown", 0)
    ranked_counts = Counter({k: v for k, v in counts.items() if k != "Unknown"})
    total_ranked = sum(ranked_counts.values())
    out = []
    for company, cnt in ranked_counts.most_common():
        out.append({"company": company, "ticket_count": cnt, "share_pct": round((cnt / total_ranked) * 100, 1) if total_ranked else 0.0})
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


def build_institution_trend_map(trends_json: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
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
        out[inst.lower()] = (patterns_text + cite).strip() or ("No specific pattern provided" + cite)
    return out


def build_company_markdown_table(
    company_counts: list[dict[str, Any]],
    institution_trend_map: dict[str, str],
    limit: int = 15,
) -> str:
    lines = [
        "| Rank | Institution | Tickets | Share | Institutional Trends |",
        "|---:|---|---:|---:|---|",
    ]
    for i, row in enumerate(company_counts[:limit], start=1):
        company = str(row.get("company", ""))
        trend_text = institution_trend_map.get(company.lower(), "")
        lines.append(
            f"| {i} | {company} | {row['ticket_count']} | {row['share_pct']}% | {trend_text} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly trend report with deterministic institution counts")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--gemini-max-per-minute", type=int, default=20)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing Gemini API key. Pass --api-key or set GEMINI_API_KEY")

    rows = read_rows(Path(args.csv))
    if not rows:
        raise SystemExit("CSV has no rows")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gemini_limiter = FixedWindowRateLimiter(args.gemini_max_per_minute, 60.0)
    rows = infer_missing_categories(rows, args.api_key, args.model, gemini_limiter)

    enriched_csv = outdir / "enriched_tickets.csv"
    write_rows(enriched_csv, rows)

    company_counts, unknown_company_count = compute_company_counts(rows)
    class_code_company_counts = compute_class_code_company_counts(rows, limit=30)
    (outdir / "company_counts.json").write_text(json.dumps(company_counts, indent=2), encoding="utf-8")
    (outdir / "class_code_company_counts.json").write_text(json.dumps(class_code_company_counts, indent=2), encoding="utf-8")

    corpus = build_ticket_corpus(rows, args.max_rows)
    prompt_trends = f"""
You are analyzing support-ticket trends for universities/institutions.

Return STRICT JSON with this exact shape:
{{
  "key_trends": [
    {{"trend": "", "why_it_matters": "", "evidence_ticket_indexes": [1,2,3]}}
  ],
  "institution_friction_patterns": [
    {{"institution": "", "patterns": ["", ""], "evidence_ticket_indexes": [1,2]}}
  ],
  "data_quality_caveats": [""]
}}

Rules:
- Focus only on trend surfacing, not recommendations or action plans.
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

    trends_text = call_gemini(args.api_key, args.model, prompt_trends, gemini_limiter)
    trends_json = extract_json(trends_text)
    (outdir / "trend_insights.json").write_text(json.dumps(trends_json, indent=2), encoding="utf-8")
    institution_trend_map = build_institution_trend_map(
        trends_json if isinstance(trends_json, dict) else {},
        rows,
    )

    date_range_label = derive_date_range_label(rows)
    title = "# Weekly Ticket Trend Report"
    if date_range_label:
        title = f"# Weekly Ticket Trend Report ({date_range_label})"

    md_parts = [
        title,
        "",
        "## Top Institutions By Ticket Volume",
        build_company_markdown_table(company_counts, institution_trend_map, limit=20),
        "",
        "## Top Class Codes By Institution",
    ]
    md_parts.extend(render_class_code_lines(class_code_company_counts[:20]))
    md_parts.append("")

    md_parts.extend(render_trend_text_sections(trends_json if isinstance(trends_json, dict) else {}, rows))
    (outdir / "final_report.md").write_text("\n".join(md_parts), encoding="utf-8")

    print(f"Wrote report artifacts to: {outdir}")
    print(f"Enriched CSV: {enriched_csv}")
    print(f"Company counts: {outdir / 'company_counts.json'}")
    print(f"Class-code company counts: {outdir / 'class_code_company_counts.json'}")
    print(f"Trend insights: {outdir / 'trend_insights.json'}")
    print(f"Final report: {outdir / 'final_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
