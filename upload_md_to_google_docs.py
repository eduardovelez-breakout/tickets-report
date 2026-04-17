#!/usr/bin/env python3
"""Upload a markdown file to Google Docs and print the Doc URL."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

SCOPES = [
    "https://www.googleapis.com/auth/drive",
]


def csv_date_range(csv_path: Path) -> str:
    if not csv_path.exists():
        return ""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="~")
        rows = [r for r in reader]
    if len(rows) < 2:
        return ""
    header = rows[0]
    created_idx = header.index("Created") if "Created" in header else 0
    dates: list[str] = []
    for row in rows[1:]:
        if len(row) <= created_idx:
            continue
        created = str(row[created_idx]).strip()
        if not created:
            continue
        date_part = created.split("T", 1)[0]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_part):
            dates.append(date_part)
    if not dates:
        return ""
    return f"{min(dates)} to {max(dates)}"


def get_or_create_folder_id(drive, folder_name: str) -> str:
    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"name = '{safe_name}' and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    found = drive.files().list(q=query, fields="files(id,name)", pageSize=10).execute()
    files = found.get("files", [])
    if files:
        return str(files[0].get("id"))
    created = drive.files().create(
        body={
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        },
        fields="id",
    ).execute()
    folder_id = str(created.get("id") or "")
    if not folder_id:
        raise RuntimeError("Failed to create/get target Drive folder")
    return folder_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload markdown file to Google Docs")
    parser.add_argument("--file", default="report_artifacts/final_report.md")
    parser.add_argument("--csv-path", default="Tickets - Last 7 Days.csv")
    parser.add_argument("--credentials", default="google_service_account.json")
    parser.add_argument("--title", default="")
    parser.add_argument("--title-prefix", default="Weekly Ticket Report")
    parser.add_argument("--folder-name", default="Weekly Ticket Reports")
    parser.add_argument("--share-anyone-read", action="store_true")
    parser.add_argument("--share-domain", default="", help="Google Workspace domain to share with (e.g. breakoutlearning.com)")
    parser.add_argument("--share-role", default="writer", choices=["reader", "commenter", "writer"], help="Role for --share-domain")
    args = parser.parse_args()

    md_path = Path(args.file)
    if not md_path.exists():
        raise SystemExit(f"File not found: {md_path}")

    content = md_path.read_text(encoding="utf-8")
    title = args.title.strip()
    if not title:
        date_range = csv_date_range(Path(args.csv_path))
        title = f"{args.title_prefix.strip() or 'Weekly Ticket Report'} ({date_range})" if date_range else (args.title_prefix.strip() or md_path.stem)

    creds = service_account.Credentials.from_service_account_file(args.credentials, scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds)
    folder_id = get_or_create_folder_id(drive, args.folder_name.strip() or "Weekly Ticket Reports")

    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/markdown", resumable=False)
    file_meta = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }

    created = drive.files().create(body=file_meta, media_body=media, fields="id,webViewLink").execute()
    file_id = str(created.get("id") or "")
    if not file_id:
        raise RuntimeError("Google Docs create succeeded but no file id returned")

    if args.share_anyone_read:
        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

    if args.share_domain:
        drive.permissions().create(
            fileId=file_id,
            body={"type": "domain", "role": args.share_role, "domain": args.share_domain},
        ).execute()

    url = str(created.get("webViewLink") or f"https://docs.google.com/document/d/{file_id}/edit")
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
