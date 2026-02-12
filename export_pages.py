#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv
from bookstack_client import BookStackClient



def main():
    load_dotenv(".env")
    out_dir = 'exports'
    token_id = os.getenv("BOOKSTACK_TOKEN_ID")
    token_secret = os.getenv("BOOKSTACK_TOKEN_SECRET")
    base_url = os.getenv("BOOKSTACK_URL")
    if not token_id or not token_secret or not base_url:
        print("Missing BOOKSTACK_TOKEN_ID or BOOKSTACK_TOKEN_SECRET or BOOKSTACK_URL", file=sys.stderr)
        return 1

    
    client = BookStackClient(base_url, token_id, token_secret)

    os.makedirs(out_dir, exist_ok=True)
    try:
        payload = client.list_pages()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    for item in payload.get("data", []):
        page_id = item.get("id")
        if page_id is None:
            continue
        try:
            pdf = client.export_page_pdf(page_id)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            continue
        out_file = os.path.join(out_dir, f"page-{page_id}.pdf")
        with open(out_file, "wb") as f:
            f.write(pdf)
        print(f"Saved {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
