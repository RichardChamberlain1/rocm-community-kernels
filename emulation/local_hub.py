"""Emulated Hugging Face Hub for kernels.

"Publishing" here writes a kernel's build output into a local directory laid
out like the Hugging Face Hub cache that the `kernels` library reads in
offline mode (`HF_HUB_OFFLINE=1`, `KERNELS_CACHE=<hub>`):

    <hub>/kernels--<org>--<name>/
        refs/<branch>                      # file holding the commit id
        snapshots/<commit>/build/<variant>/...

`get_kernel("<org>/<name>", version=N)` then resolves `refs/vN`, picks the
build variant for the machine and imports it, exactly as it does after a real
download. Nothing is ever sent to the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class LocalHubError(Exception):
    pass


@dataclass
class PushResult:
    repo_id: str
    branch: str
    commit: str
    snapshot: str
    variants: list[str]
    emulated: bool = True
    extra_files: list[str] = field(default_factory=list)


def repo_folder(hub: Path, repo_id: str) -> Path:
    """The cache folder for a kernel repo, as `huggingface_hub` names it."""
    _check_repo_id(repo_id)
    return hub / ("kernels--" + repo_id.replace("/", "--"))


def build_variants(build: Path) -> list[str]:
    """Variant folders in a kernel-builder output directory (each has metadata.json)."""
    if not build.is_dir():
        raise LocalHubError(f"build directory not found: {build}")
    variants = sorted(
        p.name for p in build.iterdir() if (p / "metadata.json").is_file()
    )
    if not variants:
        raise LocalHubError(
            f"no build variants (no <variant>/metadata.json) in {build}"
        )
    return variants


def tree_commit(build: Path, extra_files: list[Path] | None = None) -> str:
    """A deterministic 40-hex commit id for the content being published."""
    h = hashlib.sha1()
    entries = []
    for path in sorted(build.rglob("*")):
        if path.is_file():
            entries.append(("build/" + path.relative_to(build).as_posix(), path))
    for path in extra_files or []:
        entries.append((path.name, path))
    for name, path in entries:
        h.update(name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def push(
    hub: Path,
    repo_id: str,
    build: Path,
    branch: str,
    extra_files: list[Path] | None = None,
) -> PushResult:
    """Emulate `kernel-builder upload`: copy the build into a snapshot and point the branch at it."""
    _check_repo_id(repo_id)
    if not BRANCH.match(branch) or ".." in branch:
        raise LocalHubError(f"invalid branch name: {branch!r}")
    variants = build_variants(build)
    extra_files = [Path(p) for p in (extra_files or []) if Path(p).is_file()]
    commit = tree_commit(build, extra_files)

    folder = repo_folder(hub, repo_id)
    snapshot = folder / "snapshots" / commit
    if not snapshot.exists():
        tmp = snapshot.with_name(commit + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        # symlinks=False: copy file contents, as an upload would (HF cache snapshots are symlinks).
        shutil.copytree(build, tmp / "build", symlinks=False)
        for path in extra_files:
            shutil.copy2(path, tmp / path.name)
        tmp.rename(snapshot)

    ref = folder / "refs" / branch
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)  # no newline, as huggingface_hub writes refs
    return PushResult(
        repo_id,
        branch,
        commit,
        str(snapshot),
        variants,
        extra_files=[p.name for p in extra_files],
    )


def refs(hub: Path, repo_id: str) -> dict[str, str]:
    root = repo_folder(hub, repo_id) / "refs"
    if not root.is_dir():
        return {}
    return {
        p.relative_to(root).as_posix(): p.read_text().strip()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def resolve(hub: Path, repo_id: str, ref: str) -> Path:
    """Snapshot directory for a branch or commit."""
    folder = repo_folder(hub, repo_id)
    commit = ref if re.fullmatch(r"[0-9a-f]{40}", ref) else refs(hub, repo_id).get(ref)
    if commit is None:
        raise LocalHubError(f"{repo_id}: no ref {ref!r} in {folder}")
    snapshot = folder / "snapshots" / commit
    if not snapshot.is_dir():
        raise LocalHubError(
            f"{repo_id}: ref {ref!r} points at missing snapshot {commit}"
        )
    return snapshot


def layout(hub: Path, repo_id: str) -> dict:
    """A comparable description of a repo's cache layout (used by the self-test)."""
    out = {}
    for name, commit in refs(hub, repo_id).items():
        snap = repo_folder(hub, repo_id) / "snapshots" / commit
        build = snap / "build"
        out[name] = {
            "snapshot_exists": snap.is_dir(),
            "variants": (
                sorted(
                    p.name for p in build.iterdir() if (p / "metadata.json").is_file()
                )
                if build.is_dir()
                else []
            ),
        }
    return out


def _check_repo_id(repo_id: str) -> None:
    if not REPO_ID.match(repo_id):
        raise LocalHubError(f"invalid repo id: {repo_id!r} (expected <org>/<name>)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="publish a build directory to the emulated Hub")
    p.add_argument("--hub", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--extra-file", type=Path, action="append", default=[])
    r = sub.add_parser("refs", help="list a repo's branches")
    r.add_argument("--hub", type=Path, required=True)
    r.add_argument("--repo-id", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "push":
            print(
                json.dumps(
                    asdict(
                        push(
                            args.hub,
                            args.repo_id,
                            args.build,
                            args.branch,
                            args.extra_file,
                        )
                    ),
                    indent=2,
                )
            )
        else:
            print(json.dumps(refs(args.hub, args.repo_id), indent=2))
    except LocalHubError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
