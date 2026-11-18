"""Real Hub publishing for the CI emulation: private, POC organisation only.

The only file in the emulation (and scripts/) that uploads to the Hugging
Face Hub. `ci_emulate.py run --hub-org rocm-community-poc` uses it to:

  upload    `kernel-builder upload ... --private`, as publish.yml does, to
            rocm-community-poc/<name> (build.toml's organisation is swapped by
            scripts/publish_target.py, the name is kept)
  check     the repository is private after the upload: `--private` only
            applies when the upload creates the repository, so a pre-existing
            public one would otherwise stay public

It never deletes anything: emulation/hub_delete.py, run by hand, does that.

Every function refuses any organisation other than ALLOWED_ORG, and names
with a prefix the community rules reserve for AMD. The token is passed in by
the caller (from the HF_TOKEN environment variable); it is sent only in the
Authorization header or a child's environment, never printed or written.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

EMULATION = Path(__file__).resolve().parent
sys.path.insert(0, str(EMULATION.parent / "scripts"))

import check_wrapping  # noqa: E402
import publish_target  # noqa: E402
from kernel_ci import KERNEL_BUILDER  # noqa: E402

ALLOWED_ORG = "rocm-community-poc"
HUB = "https://huggingface.co"


def check_repo_id(repo_id: str) -> None:
    """Refuse anything but <ALLOWED_ORG>/<name>, and names reserved for AMD."""
    org, _, name = repo_id.partition("/")
    if org != ALLOWED_ORG or not name or "/" in name:
        raise ValueError(
            f"refusing {repo_id!r}: the emulation only publishes to {ALLOWED_ORG}/<name>"
        )
    prefixes = check_wrapping.load_community_rules()["reserved-prefixes"]
    if any(name.lower().startswith(p) for p in prefixes):
        raise ValueError(f"refusing {repo_id!r}: the name is reserved for AMD")


def target(kernel_dir: Path, hub_org: str, branch: str = "") -> dict:
    """Where publish.yml would upload this kernel (scripts/publish_target.py)."""
    if hub_org != ALLOWED_ORG:
        raise ValueError(
            f"refusing --hub-org {hub_org!r}: the emulation only publishes to {ALLOWED_ORG}"
        )
    kernel_dir = kernel_dir.resolve()
    t = publish_target.target(kernel_dir.parent, kernel_dir.name, hub_org, branch)
    check_repo_id(t["repo-id"])
    return t


def upload_cmd(kernel_dir: Path, repo_id: str, branch: str, out_json: Path) -> list:
    """The upload command: always --private. kernel_dir holds the build/ to upload."""
    check_repo_id(repo_id)
    return [
        *KERNEL_BUILDER,
        "upload",
        kernel_dir,
        "--repo-type",
        "kernel",
        "--repo-id",
        repo_id,
        "--branch",
        branch,
        "--private",
        "--output-json",
        out_json,
    ]


def request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    """A Hub API call with the token in the Authorization header; the JSON answer."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode()
    return json.loads(text) if text.strip() else {}


def is_private(repo_id: str, token: str) -> bool:
    """Whether the kernel repository is private (read with the token)."""
    check_repo_id(repo_id)
    return request("GET", f"{HUB}/api/kernels/{repo_id}", token).get("private") is True
