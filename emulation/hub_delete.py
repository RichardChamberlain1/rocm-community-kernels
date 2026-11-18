#!/usr/bin/env python3
"""Inspect, then delete, kernel repositories the CI emulation published.

Run by hand only; nothing else in the repository deletes from the Hub. Only
repositories in rocm-community-poc can be named, and not names the community
rules reserve for AMD.

Without --delete, it only shows each repository: its Hub page (look at it
there before deleting), whether it's private, its branches and when it last
changed. With --delete, it shows the same, then asks you to type the full
repository id before deleting each one. Deleting is irreversible.

  python emulation/hub_delete.py --hub-org rocm-community-poc scale-triton
  python emulation/hub_delete.py --hub-org rocm-community-poc scale-triton --delete

The token comes from HF_TOKEN (read access to look, admin rights in the
organisation to delete); set it with `read -rs HF_TOKEN && export HF_TOKEN`.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hub_publish import ALLOWED_ORG, HUB, check_repo_id, request  # noqa: E402


def describe(repo_id: str, token: str) -> dict | None:
    """The repository's page, visibility, branches and last change; None if missing."""
    try:
        info = request("GET", f"{HUB}/api/kernels/{repo_id}", token)
        refs = request("GET", f"{HUB}/api/kernels/{repo_id}/refs", token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):  # the Hub answers 401 for private repos it won't show
            return None
        raise
    return {
        "url": f"{HUB}/kernels/{repo_id}",
        "private": info.get("private"),
        "last_modified": info.get("lastModified"),
        "branches": sorted(b["name"] for b in refs.get("branches", [])),
    }


def delete(repo_id: str, token: str) -> None:
    """Delete the repository. Irreversible."""
    check_repo_id(repo_id)
    org, name = repo_id.split("/")
    request(
        "DELETE",
        f"{HUB}/api/repos/delete",
        token,
        {"type": "kernel", "organization": org, "name": name},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="+", help="repository names, e.g. scale-triton")
    ap.add_argument("--hub-org", required=True, help=f"must be {ALLOWED_ORG}")
    ap.add_argument(
        "--delete",
        action="store_true",
        help="delete each repository after you type its full id to confirm",
    )
    args = ap.parse_args(argv)
    if args.hub_org != ALLOWED_ORG:
        sys.exit(f"refusing --hub-org {args.hub_org!r}: only {ALLOWED_ORG}")
    repos = [f"{args.hub_org}/{name}" for name in args.names]
    try:
        for repo in repos:
            check_repo_id(repo)
    except ValueError as e:
        sys.exit(str(e))
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN without saving it: read -rs HF_TOKEN && export HF_TOKEN")

    failed = 0
    for repo in repos:
        d = describe(repo, token)
        if d is None:
            print(f"{repo}: not found (or this token can't see it)\n")
            continue
        print(f"{repo}\n  page:          {d['url']}")
        print(f"  private:       {d['private']}")
        print(f"  branches:      {', '.join(d['branches']) or '(none)'}")
        print(f"  last modified: {d['last_modified']}")
        if not args.delete:
            print()
            continue
        try:
            typed = input(f"  type {repo} to delete it (anything else keeps it): ")
        except EOFError:
            typed = ""
        if typed.strip() != repo:
            print("  kept\n")
            continue
        try:
            delete(repo, token)
            print("  deleted\n")
        except urllib.error.HTTPError as e:
            failed += 1
            print(f"  could not delete: HTTP {e.code} {e.reason}\n")
    if not args.delete:
        print("Nothing deleted. Inspect the pages above, then add --delete.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
