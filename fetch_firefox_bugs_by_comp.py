#!/usr/bin/env python3
import requests
import json
import argparse
import time
import os
from collections import defaultdict

API_URL = "https://bugzilla.mozilla.org/rest/bug"

def fetch_bugs(n: int, outdir: str, delay: float = 0.5):
    """
    Fetch the latest N bug reports for Firefox from Bugzilla API.
    Save as JSON files split by component.
    """
    params = {
        "product": "Firefox",
        "include_fields": "id,type,summary,component",
        "order": "bug_id DESC",
        "limit": 100
    }

    bugs_by_comp = defaultdict(list)

    os.makedirs(outdir, exist_ok=True)

    total = 0
    for offset in range(0, n, params["limit"]):
        params["offset"] = offset
        resp = requests.get(API_URL, params=params)
        if resp.status_code != 200:
            print(f"[error] HTTP {resp.status_code} at offset {offset}")
            break

        data = resp.json()
        chunk = data.get("bugs", [])
        if not chunk:
            print("[done] No more bugs found.")
            break

        for bug in chunk:
            comp = bug.get("component", "Unknown").replace("/", "_")
            bugs_by_comp[comp].append({
                "id": bug.get("id"),
                "type": bug.get("type", ""),
                "summary": bug.get("summary", ""),
                "comp": bug.get("component", ""),
                "product": "Firefox"
            })

        total += len(chunk)
        print(f"[ok] fetched {total}/{n}")
        time.sleep(delay)

    # save per component
    for comp, bugs in bugs_by_comp.items():
        outfile = os.path.join(outdir, f"{comp}.json")
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(bugs, f, ensure_ascii=False, indent=2)
        print(f"[save] {comp}: {len(bugs)} bugs -> {outfile}")

    print(f"[done] total saved: {total} bugs into {len(bugs_by_comp)} component files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest Firefox bugs from Bugzilla, split by component")
    parser.add_argument("--n", type=int, default=1000, help="Number of bugs to fetch")
    parser.add_argument("--outdir", type=str, default="firefox_bugs_by_comp", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay (s) between requests")

    args = parser.parse_args()
    fetch_bugs(args.n, args.outdir, args.delay)
