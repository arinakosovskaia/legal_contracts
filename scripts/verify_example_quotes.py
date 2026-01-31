#!/usr/bin/env python3
"""
Verify that analyzing the example contract (LIMEENERGYCO) yields findings
that contain all 4 expected quote prefixes.

Usage (from repo root, with server running and auth):
  python3 scripts/verify_example_quotes.py [--base-url http://localhost:8000] [--user demo] [--password demo]

Or in Docker:
  docker compose exec web python3 /app/scripts/verify_example_quotes.py
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Prefixes (or full text) that must appear in findings for the example contract.
EXPECTED_QUOTES = [
    ("1", "If Company terminates the Agreement without cause and for reasons other than Distributor's failure to meet its minimum expectations"),
    ("2", "any price changes, other than those based on the CPI, shall be uniformly applied to all distributors of the Products"),
    ("3", "Company and Distributor agree to indemnify, defend and hold each other harmless from any and all suits, claims, obligations, liabilities, damages, losses and the like"),
    ("4", "any and all suits, claims, obligations, liabilities, damages, losses and the like (including attorneys' fees and costs) relating to or arising out of"),
]


def _req(url: str, data: dict | None = None, auth: tuple[str, str] | None = None, timeout: int = 60) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="POST" if data else "GET")
    if auth:
        b64 = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {b64}")
    if data:
        body = urllib.request.urlencode(data).encode()
        req.data = body
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify example contract findings")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--user", default="demo", help="Basic auth user")
    parser.add_argument("--password", default="demo", help="Basic auth password")
    parser.add_argument("--timeout", type=int, default=300, help="Max seconds to wait for job")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    auth = (args.user, args.password)

    print("Starting example contract job (/upload-example)...")
    try:
        status, data = _req(
            f"{base}/upload-example",
            data={
                "provider": "nebius",
                "api_key": "",
                "base_url": "https://api.tokenfactory.nebius.com/v1/",
                "model": "",
                "use_demo_key": "1",
            },
            auth=auth,
        )
    except urllib.error.HTTPError as e:
        print(f"Upload failed: {e.code} {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Upload error: {e}", file=sys.stderr)
        return 1
    if status != 200:
        print(f"Upload returned {status}", file=sys.stderr)
        return 1
    job_id = data.get("job_id")
    if not job_id:
        print("No job_id in response", file=sys.stderr)
        return 1
    print(f"Job ID: {job_id}. Waiting for completion (timeout={args.timeout}s)...")

    deadline = time.time() + args.timeout
    result = None
    while time.time() < deadline:
        try:
            status, result = _req(f"{base}/job/{job_id}/result", auth=auth, timeout=10)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                time.sleep(3)
                continue
            raise
        if status == 200:
            break
        time.sleep(3)
    else:
        print("Job did not complete in time.", file=sys.stderr)
        return 1

    findings = (result or {}).get("findings", [])

    # Match: expected prefix contained in finding quote, or finding quote contained in expected (short overlap ok)
    quotes_lower = [(qid, qtext.strip().lower()) for qid, qtext in EXPECTED_QUOTES]
    found = []
    missing = []
    for qid, qtext_lower in quotes_lower:
        for f in findings:
            eq = (f.get("evidence_quote") or "").strip().lower()
            if not eq:
                continue
            # Accept if expected prefix is in finding, or first 60 chars of expected are in finding
            if qtext_lower in eq:
                found.append((qid, f.get("category_name", "")))
                break
            if len(qtext_lower) >= 50 and qtext_lower[:50] in eq:
                found.append((qid, f.get("category_name", "")))
                break
            if len(eq) >= 40 and eq[:40] in qtext_lower:
                found.append((qid, f.get("category_name", "")))
                break
        else:
            missing.append(qid)

    if not missing:
        print("OK: All 4 expected quote prefixes appear in findings.")
        for qid, cat in found:
            print(f"  Quote {qid} -> {cat}")
        return 0

    print("MISSING: The following quote prefixes were not found in any finding:", file=sys.stderr)
    for qid in missing:
        for (eq_id, text) in EXPECTED_QUOTES:
            if eq_id == qid:
                print(f"  Quote {qid}: {text[:80]}...", file=sys.stderr)
                break
    print(f"Total findings: {len(findings)}. Categories: {[f.get('category_name') for f in findings]}", file=sys.stderr)
    print("Tip: Run ./scripts/refresh_example_contract.sh so example is LIMEENERGYCO; set LLM_TEMPERATURE=0 and LLM_SEED=42.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

