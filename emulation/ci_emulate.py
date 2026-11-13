#!/usr/bin/env python3
"""Emulate the rocm-community-kernels CI locally, including publishing.

Runs the pull request CI stages from scripts/kernel_ci.py, then emulates what
happens after an approved publish:

  validate   check it is wrapped correctly for Hugging Face's kernel-builder
  build      nix build .#backendBundle.<backend> (all variants), check-abi;
             or --build-dir to use a prebuilt build/
  test       nix build .#ci-test and run it on the AMD GPU (pytest -m kernels_ci)
  publish    EMULATED: copy the build into a local Hub (emulation/local_hub.py)
  verify     get_kernel() from the emulated Hub in a user-like Python env,
             then the kernel's own kernels_ci tests against that load
  benchmark  time benchmarks/benchmark.py from the emulated Hub (perf.json)

for the kernels in a list (emulation/approved-kernels.toml by default, or
emulation/test-kernels.toml) or given with --kernel, and writes
ci-out/<run-id>/summary.{json,md}. Nothing is ever pushed to the Hugging Face
Hub: no stage calls an upload, Hugging Face tokens are removed from every
command's environment, and every stage after the self-test's download runs
with HF_HUB_OFFLINE=1.

`selftest` checks the emulation against a real Hub kernel: it downloads one
(read-only), re-publishes the same files to the emulated Hub, and checks that
get_kernel() picks the same variant and gives the same result from both.

Everything in emulation/ goes once the real CI publishes (.github/workflows/publish.yml).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

EMULATION = Path(__file__).resolve().parent
REPO_ROOT = EMULATION.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EMULATION))

import kernel_ci  # noqa: E402
import local_hub  # noqa: E402
from kernel_ci import (  # noqa: E402
    KERNEL_BUILDER,
    KernelRun,
    StageResult,
    arch_mismatch,
    env_for,
    git_sha,
    gpu_info,
)

STAGES = kernel_ci.STAGES + ["publish", "verify", "benchmark"]
VERIFY = SCRIPTS / "ci_verify_load.py"
DEFAULT_APPROVED = EMULATION / "approved-kernels.toml"
RELU_SMOKE = "k.relu(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=device))"


def load_entries(args) -> list[dict]:
    """Kernels from the list file (approved list or a test-case list) plus --kernel kernels.

    The approved list is only read by default when no --kernel is given. A list may
    set `kernels_root` (relative to the list file); --kernels-root overrides it.
    """
    entries = []
    default_root = args.kernels_root or REPO_ROOT
    if args.approved is None and not args.kernel:
        args.approved = DEFAULT_APPROVED
    if args.approved and args.approved.is_file():
        with open(args.approved, "rb") as f:
            doc = tomllib.load(f)
        root = args.kernels_root or (
            (args.approved.parent / doc["kernels_root"]).resolve()
            if "kernels_root" in doc
            else REPO_ROOT
        )
        approved = args.approved.resolve() == DEFAULT_APPROVED.resolve()
        label = "" if approved else f"  (test case from {args.approved.name})"
        for e in doc.get("kernel", []):
            if "path" not in e:
                sys.exit(f"{args.approved}: every [[kernel]] needs a path")
            entries.append({"approved": approved, "_root": root, "_label": label, **e})
    for k in args.kernel:  # ad hoc trials, outside the approved list
        entries.append({"path": k, "_root": default_root, "_label": "  (trial)"})
    if args.only:
        entries = [e for e in entries if Path(e["path"]).name in args.only]
    return entries


# --------------------------------------------------------------------------- stages


def stage_build(ctx, kr: KernelRun) -> StageResult:
    """kernel_ci's build, or --build-dir to use a prebuilt build/ instead of Nix."""
    if not ctx.args.build_dir:
        if not ctx.env["nix"] and not ctx.dry_run:
            return StageResult(
                "skip",
                message="nix not found; pass --build-dir to use a prebuilt build/",
            )
        return kernel_ci.stage_build(ctx, kr)
    build = kr.out / "build"
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(ctx.args.build_dir, build, symlinks=False)
    return StageResult(
        "pass",
        message="prebuilt build directory",
        details={"variants": local_hub.build_variants(build)},
    )


def publish_branch(ctx, kr: KernelRun) -> str:
    return kr.entry.get("branch") or ctx.args.branch or f"v{kr.info.version}"


def stage_publish(ctx, kr: KernelRun) -> StageResult:
    build = kr.out / "build"
    if (
        not build.is_dir() and ctx.args.build_dir
    ):  # build stage not run: publish the prebuilt directory
        build = ctx.args.build_dir.resolve()
    if ctx.dry_run:
        print(
            f"    [emulated] local_hub.push {kr.info.repo_id} -> {ctx.hub} branch {publish_branch(ctx, kr)}"
        )
        return StageResult("dry-run")
    if not build.is_dir():
        return StageResult("skip", message="no build to publish")
    if ctx.env["nix"] and (kr.kernel_dir / "CARD.md").is_file():
        ctx.run(
            [*KERNEL_BUILDER, "fill-card", kr.kernel_dir, build / "CARD.md"],
            kr.out / "publish.log",
            cwd=kr.kernel_dir,
        )
    branch = publish_branch(ctx, kr)
    extra = [kr.kernel_dir / "README.md"]
    result = local_hub.push(ctx.hub, kr.info.repo_id, build, branch, extra)
    branches = [branch]
    # Version-1 kernels without a branch override also go to main.
    if kr.info.version == 1 and not (kr.entry.get("branch") or ctx.args.branch):
        local_hub.push(ctx.hub, kr.info.repo_id, build, "main", extra)
        branches.append("main")
    record = {
        **asdict(result),
        "branches": branches,
        "signed": False,
        "note": "emulated; nothing was sent to the Hub",
    }
    (kr.out / "upload.json").write_text(json.dumps(record, indent=2) + "\n")
    return StageResult(
        "pass",
        message=f"{kr.info.repo_id}@{'/'.join(branches)} = {result.commit[:12]}",
        details=record,
    )


def _ref_args(ctx, kr: KernelRun) -> list[str]:
    branch = publish_branch(ctx, kr)
    return (
        ["--version", str(kr.info.version)]
        if branch == f"v{kr.info.version}"
        else ["--revision", branch]
    )


def stage_verify(ctx, kr: KernelRun) -> StageResult:
    if why := arch_mismatch(ctx, kr):
        return StageResult("skip", message=why)
    if (
        not ctx.dry_run
        and kr.results.get("publish", StageResult("skip")).status != "pass"
    ):
        return StageResult("skip", message="nothing published")
    report = kr.out / "verify.json"
    tests = Path(kr.info.config_dir) / "tests"
    cmd = [
        ctx.args.python,
        VERIFY,
        "--repo-id",
        kr.info.repo_id,
        *_ref_args(ctx, kr),
        "--expect-under",
        local_hub.repo_folder(ctx.hub, kr.info.repo_id),
        "--json",
        report,
    ]
    if tests.is_dir() and not ctx.args.no_tests:
        cmd += ["--tests", tests]
    rc, secs = ctx.run(
        cmd,
        kr.out / "verify.log",
        cwd=Path(kr.info.config_dir),
        env=env_for(True, KERNELS_CACHE=str(ctx.hub)),
    )
    if ctx.dry_run:
        return StageResult("dry-run")
    details = json.loads(report.read_text()) if report.is_file() else {}
    msg = (
        f"variant {details.get('variant')}"
        if rc == 0
        else (details.get("error") or "see verify.log")
    )
    if details.get("digest_warnings"):
        msg += "; digest warning(s)"
    return StageResult("pass" if rc == 0 else "fail", message=msg, details=details)


def stage_benchmark(ctx, kr: KernelRun) -> StageResult:
    if why := arch_mismatch(ctx, kr):
        return StageResult("skip", message=why)
    if kr.entry.get("benchmark", True) is False:
        return StageResult("skip", message="benchmark = false in the approved list")
    bench = next(
        (
            p
            for p in (
                Path(kr.info.config_dir) / "benchmarks" / "benchmark.py",
                kr.kernel_dir / "benchmarks" / "benchmark.py",
            )
            if p.is_file()
        ),
        None,
    )
    if bench is None:
        return StageResult("skip", message="no benchmarks/benchmark.py")
    if (
        not ctx.dry_run
        and kr.results.get("publish", StageResult("skip")).status != "pass"
    ):
        return StageResult("skip", message="nothing published")
    report = kr.out / "benchmark.json"
    cmd = [
        ctx.args.python,
        VERIFY,
        "--repo-id",
        kr.info.repo_id,
        *_ref_args(ctx, kr),
        "--expect-under",
        local_hub.repo_folder(ctx.hub, kr.info.repo_id),
        "--benchmark",
        bench,
        "--iters",
        str(ctx.args.iters),
        "--json",
        report,
    ]
    rc, _ = ctx.run(
        cmd,
        kr.out / "benchmark.log",
        cwd=Path(kr.info.config_dir),
        env=env_for(True, KERNELS_CACHE=str(ctx.hub)),
    )
    if ctx.dry_run:
        return StageResult("dry-run")
    data = json.loads(report.read_text()) if report.is_file() else {}
    perf = {
        "kernel": kr.name,
        "repo_id": kr.info.repo_id,
        "run_id": ctx.run_id,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(kr.kernel_dir),
        "host": ctx.env["host"],
        "gpu": gpu_info(),
        **{
            k: data.get(k)
            for k in ("device", "torch", "hip", "kernels", "variant", "commit")
        },
        "results": data.get("benchmarks", []),
    }
    regressions = (
        compare_baseline(perf, ctx.args.baseline, ctx.args.threshold)
        if ctx.args.baseline
        else []
    )
    perf["regressions"] = regressions
    (kr.out / "perf.json").write_text(json.dumps(perf, indent=2) + "\n")
    if rc != 0:
        return StageResult(
            "fail",
            message=data.get("error") or "benchmark failed or did not verify",
            details=perf,
        )
    msg = f"{len(perf['results'])} workload(s)"
    if regressions:
        msg += (
            f"; {len(regressions)} slower than baseline by > {ctx.args.threshold:.0%}"
        )
        if ctx.args.fail_on_regression:
            return StageResult("fail", message=msg, details=perf)
    return StageResult("pass", message=msg, details=perf)


def compare_baseline(perf: dict, baseline: Path, threshold: float) -> list[dict]:
    """Workloads whose median got slower than the baseline perf.json by more than threshold."""
    base = json.loads(baseline.read_text())
    if base.get("kernel") not in (None, perf["kernel"]):
        base = next(
            (b for b in base.get("kernels", []) if b.get("kernel") == perf["kernel"]),
            {},
        )
    old = {
        (r.get("class"), r.get("workload")): r.get("median_ms")
        for r in base.get("results", [])
    }
    out = []
    for r in perf["results"]:
        prev = old.get((r.get("class"), r.get("workload")))
        if prev and r.get("median_ms") and r["median_ms"] > prev * (1 + threshold):
            out.append(
                {
                    "class": r["class"],
                    "workload": r["workload"],
                    "baseline_ms": prev,
                    "median_ms": r["median_ms"],
                }
            )
    return out


STAGE_FUNCS = {
    **kernel_ci.STAGE_FUNCS,
    "build": stage_build,
    "publish": stage_publish,
    "verify": stage_verify,
    "benchmark": stage_benchmark,
}


class Ctx(kernel_ci.Ctx):
    def __init__(self, args):
        super().__init__(args)
        self.hub = (args.hub or self.out / "hub").resolve()


def cmd_run(args) -> int:
    ctx = Ctx(args)
    stages = kernel_ci.select_stages(args, STAGES)
    entries = load_entries(args)
    if not entries:
        print(
            f"No kernels to run (approved list: {args.approved}; use --kernel to try one)."
        )
        return 0
    if args.build_dir and len(entries) != 1:
        sys.exit("--build-dir needs exactly one kernel")
    print(f"emulated hub {ctx.hub}")
    runs = kernel_ci.run_kernels(ctx, entries, stages, STAGE_FUNCS)
    return kernel_ci.write_report(
        ctx,
        runs,
        stages,
        title="CI emulation",
        notes=[
            f"Emulated Hub: `{ctx.hub}` (nothing was pushed to the Hugging Face Hub)."
        ],
        extra={"emulated_hub": str(ctx.hub)},
    )


def cmd_selftest(args) -> int:
    """Download a real Hub kernel, re-publish it to the emulated Hub, and compare the two loads."""
    ctx = Ctx(args)
    st = ctx.out / "selftest"
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    real_cache, hub = st / "real-cache", st / "hub"
    smoke = (
        args.smoke
        if args.smoke is not None
        else (RELU_SMOKE if args.reference.endswith("/relu") else None)
    )
    base = [
        args.python,
        VERIFY,
        "--repo-id",
        args.reference,
        "--version",
        str(args.version),
    ]
    if smoke:
        base += ["--smoke", smoke]
    print(f"selftest: {args.reference} v{args.version} -> {st}")

    print("  1. real Hub download (read-only)")
    real_cmd = [*base, "--json", st / "real.json"] + (
        ["--trust-remote-code"] if args.trust_remote_code else []
    )
    rc, _ = ctx.run(
        real_cmd, st / "selftest.log", env=env_for(False, KERNELS_CACHE=str(real_cache))
    )
    if ctx.dry_run:
        print("  2. [emulated] push the downloaded variant to", hub)
        ctx.run(
            [
                *base,
                "--expect-under",
                local_hub.repo_folder(hub, args.reference),
                "--json",
                st / "emulated.json",
            ],
            st / "selftest.log",
            env=env_for(True, KERNELS_CACHE=str(hub)),
        )
        return 0
    real = json.loads((st / "real.json").read_text())
    if rc != 0 or not real.get("variant_dir"):
        print(
            f"  FAIL: could not load {args.reference} from the Hub: {real.get('error')}"
        )
        return 1

    print("  2. re-publish the downloaded files to the emulated Hub")
    build = Path(real["variant_dir"]).parent
    pushed = local_hub.push(hub, args.reference, build, f"v{args.version}")

    print("  3. get_kernel() from the emulated Hub (offline)")
    rc, _ = ctx.run(
        [
            *base,
            "--expect-under",
            local_hub.repo_folder(hub, args.reference),
            "--json",
            st / "emulated.json",
        ],
        st / "selftest.log",
        env=env_for(True, KERNELS_CACHE=str(hub)),
    )
    emu = json.loads((st / "emulated.json").read_text())

    real_layout = local_hub.layout(real_cache, args.reference)
    emu_layout = local_hub.layout(hub, args.reference)
    ref = f"v{args.version}"
    checks = {
        "emulated load succeeded": rc == 0 and emu.get("ok"),
        "same build variant": real.get("variant") == emu.get("variant"),
        "same module file in the variant": real.get("module_relpath")
        == emu.get("module_relpath"),
        "same smoke result": (real.get("smoke") == emu.get("smoke")) if smoke else None,
        "no digest warnings (emulated)": not emu.get("digest_warnings"),
        f"refs/{ref} in the real cache": ref in real_layout,
        f"refs/{ref} in the emulated hub": ref in emu_layout,
        "same variants under the ref": real_layout.get(ref, {}).get("variants")
        == emu_layout.get(ref, {}).get("variants"),
    }
    result = {
        "reference": args.reference,
        "version": args.version,
        "pushed": asdict(pushed),
        "checks": checks,
        "real": real,
        "emulated": emu,
        "real_layout": real_layout,
        "emulated_layout": emu_layout,
    }
    (st / "selftest.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, ok in checks.items():
        print(f"     {'n/a ' if ok is None else ('PASS' if ok else 'FAIL')}  {name}")
    passed = all(v for v in checks.values() if v is not None)
    print(f"  selftest {'PASSED' if passed else 'FAILED'}: {st / 'selftest.json'}")
    return 0 if passed else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    kernel_ci.add_common_args(common)
    common.add_argument(
        "--hub", type=Path, help="emulated Hub directory (default: <out>/<run-id>/hub)"
    )
    common.add_argument(
        "--python",
        default=sys.executable,
        help="user-like Python with torch for ROCm and kernels (default: this interpreter)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "run", parents=[common], help="run the CI stages, with an emulated publish"
    )
    kernel_ci.add_run_args(r, STAGES)
    r.add_argument(
        "--approved",
        type=Path,
        default=None,
        help="kernel list: the approved list (default without --kernel) or a test-case "
        "list such as emulation/test-kernels.toml",
    )
    r.add_argument(
        "--only",
        action="append",
        default=[],
        help="only these kernels (directory names)",
    )
    r.add_argument(
        "--build-dir",
        type=Path,
        help="use this prebuilt build/ instead of building (one kernel)",
    )
    r.add_argument(
        "--branch", help="publish to this branch instead of v<version> (e.g. staging)"
    )
    r.add_argument(
        "--no-tests",
        action="store_true",
        help="verify the load only, without the kernels_ci tests",
    )
    r.add_argument("--iters", type=int, default=50)
    r.add_argument("--baseline", type=Path, help="earlier perf.json to compare against")
    r.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="slowdown that counts as a regression (0.05 = 5%%)",
    )
    r.add_argument("--fail-on-regression", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser(
        "selftest",
        parents=[common],
        help="check the emulated Hub against a real Hub kernel",
    )
    s.add_argument(
        "--reference", required=True, help="a Hub kernel to compare with, <org>/<name>"
    )
    s.add_argument("--version", type=int, default=1)
    s.add_argument(
        "--smoke",
        help="expression using k, torch, device (default for */relu: relu of a small tensor)",
    )
    s.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="for a reference outside trusted organisations",
    )
    s.set_defaults(func=cmd_selftest)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
