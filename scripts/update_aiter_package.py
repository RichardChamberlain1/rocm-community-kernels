#!/usr/bin/env python3
"""Regenerate an AMD package that builds a compiled AITER kernel from the AITER submodule.

Reads a sync/<package>.toml config and:

1. checks that the package's AITER submodule (and, for CK-based packages, AITER's
   Composable Kernel submodule) are checked out at the pinned commits;
2. for CK-based packages, regenerates the CK instances with CK's own codegen (.cpp
   renamed to .hip, so the build compiles them as HIP);
3. finds every header the sources and instances include, by following #include lines
   through the include directories (kernel-builder copies only files listed in
   build.toml into the Nix build);
4. writes the package's build.toml.

With --to <ref> it first moves the package to another AITER revision: it fetches the
ref (a release tag such as v0.1.29, or a commit) into the AITER submodule, checks out
the Composable Kernel commit that AITER revision pins (if the package uses CK), and
updates the pins in the config. Then stage the result, including the submodule: git add -A

With --check nothing is written: it exits 1 if the instances or build.toml differ
from what would be generated. Run from anywhere:

    python3 scripts/update_aiter_package.py sync/aiter-moe-ops.toml [--to v0.1.29 | --check]

The submodules must be checked out: git submodule update --init --recursive
"""

import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)


def git_head(path):
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if out.returncode != 0:
        sys.exit(
            f"{path} is not a checked-out git repository; run git submodule update --init --recursive"
        )
    return out.stdout.strip()


def gitlink(repo, subpath):
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "HEAD", subpath],
        capture_output=True,
        text=True,
    )
    fields = out.stdout.split()
    return fields[2] if len(fields) >= 3 and fields[1] == "commit" else None


def git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True)


def uses_ck(cfg):
    return "ck-commit" in cfg["aiter"]


def move_to(pkg, cfg, config_path, ref):
    """Check out `ref` in the AITER submodule (and its CK pin); rewrite the config's pins."""
    aiter = pkg / cfg["aiter"]["path"]
    git(aiter, "fetch", "--quiet", "--depth", "1", "origin", ref)
    git(aiter, "-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD")
    pins = {"ref": ref, "commit": git_head(aiter)}
    if uses_ck(cfg):
        ck = pkg / cfg["aiter"]["ck-path"]
        git(
            aiter,
            "submodule",
            "update",
            "--init",
            "--quiet",
            "--depth",
            "1",
            "--",
            os.path.relpath(ck, aiter),
        )
        pins["ck-commit"] = git_head(ck)
    text = config_path.read_text()
    for key, value in pins.items():
        text, n = re.subn(
            rf'^{key} = "[^"]*"', f'{key} = "{value}"', text, count=1, flags=re.M
        )
        if n != 1:
            sys.exit(f"{config_path}: no `{key} = ...` line to update")
    config_path.write_text(text)
    old = cfg["aiter"]["commit"]
    ck_note = f", Composable Kernel {pins['ck-commit'][:12]}" if uses_ck(cfg) else ""
    print(
        f"{cfg['package']}: AITER {old[:12]} -> {pins['commit'][:12]} ({ref}){ck_note}"
    )
    return tomllib.loads(text)


def check_pins(pkg, cfg):
    aiter = pkg / cfg["aiter"]["path"]
    errors = []
    if git_head(aiter) != cfg["aiter"]["commit"]:
        errors.append(
            f"AITER submodule is at {git_head(aiter)}, config pins {cfg['aiter']['commit']}"
        )
    if not uses_ck(cfg):
        if errors:
            sys.exit("\n".join(errors))
        return
    ck = pkg / cfg["aiter"]["ck-path"]
    ck_pin = gitlink(aiter, os.path.relpath(ck, aiter))
    if ck_pin != cfg["aiter"]["ck-commit"]:
        errors.append(
            f"AITER pins Composable Kernel at {ck_pin}, config says {cfg['aiter']['ck-commit']}"
        )
    if git_head(ck) != cfg["aiter"]["ck-commit"]:
        errors.append(
            f"Composable Kernel is checked out at {git_head(ck)}, expected {cfg['aiter']['ck-commit']}"
        )
    if errors:
        sys.exit("\n".join(errors))


def generate_instances(pkg, cfg, out_dir):
    """Run CK's generator for every configured command and rename .cpp to .hip."""
    gen = pkg / cfg["instances"]["generator"]
    with tempfile.TemporaryDirectory() as tmp:
        for args in cfg["instances"]["commands"]:
            # Run from a scratch directory: the generator only needs its own directory on sys.path.
            subprocess.run(
                [sys.executable, str(gen), *args, "-o", tmp],
                check=True,
                cwd=tmp,
                stdout=subprocess.DEVNULL,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(tmp).iterdir()):
            name = f.name[: -len(".cpp")] + ".hip" if f.suffix == ".cpp" else f.name
            shutil.copyfile(f, out_dir / name)


def include_closure(pkg, roots, include_dirs, overlay):
    """All package files reachable from `roots` through #include, plus unresolved names.

    Paths are relative to the package. `overlay` maps a package-relative directory to
    where its files actually are (the instances, which --check generates elsewhere).
    """

    def actual(rel):
        for prefix, real in overlay.items():
            if rel.startswith(prefix + "/"):
                return Path(real) / rel[len(prefix) + 1 :]
        return pkg / rel

    seen, external, todo = set(), set(), list(roots)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        text = actual(f).read_text(encoding="utf-8", errors="replace")
        for kind, name in INCLUDE_RE.findall(text):
            candidates = [os.path.join(os.path.dirname(f), name)] if kind == '"' else []
            candidates += [os.path.join(d, name) for d in include_dirs]
            hit = next(
                (
                    os.path.normpath(c)
                    for c in candidates
                    if actual(os.path.normpath(c)).is_file()
                ),
                None,
            )
            if hit is None:
                external.add(name)
            elif hit.startswith(".."):
                sys.exit(f"{f} includes {name}, which resolves outside the package")
            else:
                todo.append(hit)
    return seen, external


def toml_list(key, items, indent=""):
    if not items:
        return f"{indent}{key} = []\n"
    body = "".join(f'{indent}    "{i}",\n' for i in items)
    return f"{indent}{key} = [\n{body}{indent}]\n"


def render_build_toml(cfg, files):
    g, k = cfg["general"], cfg["kernel"]
    out = [
        "# Generated by scripts/update_aiter_package.py from "
        f"sync/{cfg['package']}.toml; do not edit by hand.\n",
        "[general]\n",
        f'name = "{cfg["package"]}"\n',
        f"version = {g['version']}\n",
        "edition = 6\n",
        f'license = "{g["license"]}"\n',
        f'upstream = "{g["upstream"]}"\n',
        f'source = "{g["source"]}"\n',
        'backends = ["rocm"]\n\n',
        "[general.hub]\n",
        f'repo-id = "{g["repo-id"]}"\n\n',
        "[torch]\n",
        toml_list("src", cfg["torch"]["src"]),
        "\n",
    ]
    if "stable-abi" in cfg["torch"]:
        out += ["[torch.stable-abi]\n", f'rocm = "{cfg["torch"]["stable-abi"]}"\n\n']
    if uses_ck(cfg):
        out += [
            f"# Built from ROCm/aiter {cfg['aiter']['commit'][:12]} and its Composable Kernel "
            f"{cfg['aiter']['ck-commit'][:12]}\n",
            "# (git submodules), unchanged, plus the generated CK instances in instances/.\n",
        ]
    else:
        out.append(
            f"# Built from ROCm/aiter {cfg['aiter']['commit'][:12]} (git submodule), unchanged.\n"
        )
    out += [
        f"[kernel.{k['name']}]\n",
        'language = "cpp-rocm"\n',
        'depends = ["torch"]\n',
    ]
    out += [
        toml_list(key, k[key])
        for key in ("rocm-archs", "cxx-flags", "hip-flags")
        if key in k
    ]
    out += [toml_list("include", k["include"]), toml_list("src", files)]
    return "".join(out)


def generated_dirs(cfg):
    return [cfg["instances"]["dir"]] if "instances" in cfg else []


def build(pkg, cfg, gen_root):
    """Generate instances under gen_root; return the build.toml text and unresolved includes."""
    overlay = {d: gen_root / d for d in generated_dirs(cfg)}
    instances = []
    if "instances" in cfg:
        inst = cfg["instances"]["dir"]
        generate_instances(pkg, cfg, gen_root / inst)
        instances = [f"{inst}/{f.name}" for f in sorted((gen_root / inst).iterdir())]
    files, external = include_closure(
        pkg, cfg["kernel"]["sources"] + instances, cfg["kernel"]["include"], overlay
    )
    return render_build_toml(cfg, sorted(files)), external


def diff_trees(a, b):
    """Relative paths that differ between two directory trees."""
    if not b.is_dir():
        return [f"{b.name}/ (missing)"]
    out, cmp, stack = [], filecmp.dircmp(a, b), []
    stack.append(("", cmp))
    while stack:
        prefix, c = stack.pop()
        out += [prefix + n for n in c.left_only + c.right_only + c.diff_files]
        out += [
            prefix + n
            for n in filecmp.cmpfiles(c.left, c.right, c.common_files, shallow=False)[1]
        ]
        stack += [(prefix + n + "/", sub) for n, sub in c.subdirs.items()]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify instead of writing; exit 1 if out of date",
    )
    ap.add_argument(
        "--to",
        metavar="REF",
        help="move to this AITER release tag or commit first, updating the config's pins",
    )
    ap.add_argument(
        "--show-external",
        action="store_true",
        help="print includes resolved outside the package",
    )
    a = ap.parse_args()

    if a.check and a.to:
        sys.exit("--check and --to can't be combined")
    cfg = tomllib.loads(a.config.read_text())
    pkg = REPO / cfg["package"]
    if a.to:
        cfg = move_to(pkg, cfg, a.config, a.to)
    check_pins(pkg, cfg)

    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            text, external = build(pkg, cfg, Path(tmp))
            problems = []
            for d in generated_dirs(cfg):
                stale = diff_trees(Path(tmp) / d, pkg / d)
                if stale:
                    problems.append(
                        f"{len(stale)} files in {d}/ differ, e.g. {stale[:3]}"
                    )
            if (pkg / "build.toml").read_text() != text:
                problems.append("build.toml differs from the generated one")
        if problems:
            sys.exit(
                "out of date: "
                + "; ".join(problems)
                + f"\nrun: python3 scripts/update_aiter_package.py {a.config}"
            )
        print(f"{cfg['package']}: generated files and build.toml are up to date")
    else:
        for d in generated_dirs(cfg):
            if (pkg / d).exists():
                shutil.rmtree(pkg / d)
        text, external = build(pkg, cfg, pkg)
        (pkg / "build.toml").write_text(text)
        n_src = len(tomllib.loads(text)["kernel"][cfg["kernel"]["name"]]["src"])
        inst_note = ""
        if "instances" in cfg:
            n_inst = sum(1 for _ in (pkg / cfg["instances"]["dir"]).iterdir())
            inst_note = f"{n_inst} instances, "
        print(f"{cfg['package']}: {inst_note}build.toml lists {n_src} kernel files")
    if a.show_external:
        print("includes resolved outside the package (system, HIP, Torch):")
        for name in sorted(external):
            print("  ", name)


if __name__ == "__main__":
    main()
