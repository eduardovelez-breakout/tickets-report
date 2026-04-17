#!/usr/bin/env python3
"""Upload a markdown file to Google Docs and print the Doc URL."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

SCOPES = [
    "https://www.googleapis.com/auth/drive",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload markdown file to Google Docs")
    parser.add_argument("--file", default="report_artifacts/final_report.md")
    parser.add_argument("--credentials", default="google_service_account.json")
    parser.add_argument("--title", default="")
    parser.add_argument("--share-anyone-read", action="store_true")
    parser.add_argument("--share-domain", default="", help="Google Workspace domain to share with (e.g. breakoutlearning.com)")
    parser.add_argument("--share-role", default="writer", choices=["reader", "commenter", "writer"], help="Role for --share-domain")
    args = parser.parse_args()

    md_path = Path(args.file)
    if not md_path.exists():
        raise SystemExit(f"File not found: {md_path}")

    content = md_path.read_text(encoding="utf-8")
    title = args.title.strip() or md_path.stem

    creds = service_account.Credentials.from_service_account_file(args.credentials, scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds)

    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/markdown", resumable=False)
    file_meta = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
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
