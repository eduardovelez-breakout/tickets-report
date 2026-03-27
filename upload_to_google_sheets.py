#!/usr/bin/env python3
"""Create a new worksheet in a Google Sheet and upload CSV rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def read_csv_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="~")
        return [row for row in reader]


def unique_tab_title(base: str) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{base} {now}"[:100]


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload CSV to new Google Sheets tab")
    parser.add_argument("--sheet-id", required=True)
    parser.add_argument("--csv-path", default="Tickets - Last 7 Days.csv")
    parser.add_argument("--credentials", default="google_service_account.json")
    parser.add_argument("--tab-prefix", default="Weekly Tickets")
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.csv_path))
    if not rows:
        raise RuntimeError("CSV is empty")

    creds = service_account.Credentials.from_service_account_file(args.credentials, scopes=SCOPES)
    service = build("sheets", "v4", credentials=creds)

    title = unique_tab_title(args.tab_prefix)

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

    service.spreadsheets().batchUpdate(spreadsheetId=args.sheet_id, body=add_sheet_req).execute()

    body = {"values": rows}
    service.spreadsheets().values().update(
        spreadsheetId=args.sheet_id,
        range=f"'{title}'!A1",
        valueInputOption="USER_ENTERED",
        body=body,
    ).execute()

    print(f"Uploaded {len(rows) - 1} rows to tab: {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
