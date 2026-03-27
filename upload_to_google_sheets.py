#!/usr/bin/env python3
"""Create a new worksheet in a Google Sheet and upload CSV rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def read_csv_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="~")
        return [row for row in reader]


def unique_tab_title(rows: list[list[str]]) -> str:
    if not rows or len(rows) < 2:
        return dt.datetime.now().strftime("%Y-%m-%d")

    header = rows[0]
    created_idx = 0
    if "Created" in header:
        created_idx = header.index("Created")

    dates: list[str] = []
    for row in rows[1:]:
        if len(row) <= created_idx:
            continue
        created = row[created_idx].strip()
        if not created:
            continue
        date_part = created.split("T", 1)[0]
        if len(date_part) == 10:
            dates.append(date_part)

    if not dates:
        return dt.datetime.now().strftime("%Y-%m-%d")

    start = min(dates)
    end = max(dates)
    return f"{start} to {end}"[:100]


def first_sheet_title(service, sheet_id: str) -> str:
    meta = service.spreadsheets().get(
        spreadsheetId=sheet_id,
        fields="sheets(properties(title))",
    ).execute()
    sheets = meta.get("sheets", [])
    if not sheets:
        raise RuntimeError("Spreadsheet has no sheets")
    props = sheets[0].get("properties", {})
    title = props.get("title")
    if not title:
        raise RuntimeError("Could not determine first sheet title")
    return str(title)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload CSV to new Google Sheets tab")
    parser.add_argument("--sheet-id", required=True)
    parser.add_argument("--csv-path", default="Tickets - Last 7 Days.csv")
    parser.add_argument("--credentials", default="google_service_account.json")
    parser.add_argument("--tab-prefix", default="")
    parser.add_argument("--fallback-first-sheet", action="store_true", default=True)
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.csv_path))
    if not rows:
        raise RuntimeError("CSV is empty")

    creds = service_account.Credentials.from_service_account_file(args.credentials, scopes=SCOPES)
    service = build("sheets", "v4", credentials=creds)

    title = unique_tab_title(rows)

    add_sheet_req = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": title,
                    }
                }
            }
        ]
    }

    target_title = title
    try:
        service.spreadsheets().batchUpdate(spreadsheetId=args.sheet_id, body=add_sheet_req).execute()
    except HttpError as exc:
        msg = str(exc)
        unsupported = "This operation is not supported for this document" in msg
        if not (args.fallback_first_sheet and unsupported):
            raise
        target_title = first_sheet_title(service, args.sheet_id)
        print(
            f"addSheet unsupported for this document; writing to existing tab: {target_title}",
            file=sys.stderr,
        )

    body = {"values": rows}
    service.spreadsheets().values().update(
        spreadsheetId=args.sheet_id,
        range=f"'{target_title}'!A1",
        valueInputOption="USER_ENTERED",
        body=body,
    ).execute()

    print(f"Uploaded {len(rows) - 1} rows to tab: {target_title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
