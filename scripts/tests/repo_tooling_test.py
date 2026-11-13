"""Tests for the helpers the GitHub workflows call: changed kernels, maintainers, publish target."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import changed_kernels  # noqa: E402
import kernel_maintainers  # noqa: E402
import publish_target  # noqa: E402


def add_kernel(root: Path, name: str, version: int = 1, org: str = "rocm-community"):
    k = root / name
    k.mkdir(parents=True)
    (k / "flake.nix").write_text("{}\n")
    (k / "build.toml").write_text(
        f'[general]\nname = "{name}"\nversion = {version}\n\n'
        f'[general.hub]\nrepo-id = "{org}/{name}"\n'
    )
    return k


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_changed_kernels(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ci@example.com")
    git(tmp_path, "config", "user.name", "ci")
    add_kernel(tmp_path, "alpha")
    add_kernel(tmp_path, "beta")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("x\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "alpha" / "kernel.hip").write_text("// change\n")
    (tmp_path / "docs" / "x.md").write_text("y\n")
    (tmp_path / "README.md").write_text("top-level files are not kernels\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "change")
    files = changed_kernels.changed_files(base, "HEAD", tmp_path)
    assert changed_kernels.kernels_in(files, tmp_path) == ["alpha"]
    assert changed_kernels.all_kernels(tmp_path) == ["alpha", "beta"]


def test_explicit_kernels_must_exist(tmp_path, capsys):
    add_kernel(tmp_path, "alpha")
    root = ["--root", str(tmp_path)]
    assert changed_kernels.main([*root, "--kernels", "alpha"]) == 0
    assert json.loads(capsys.readouterr().out) == ["alpha"]
    assert changed_kernels.main([*root, "--kernels", "nope"]) == 1
    assert changed_kernels.main([*root, "--kernels", "$(evil)"]) == 1


def write_registry(root: Path, registry: dict):
    (root / ".github").mkdir(exist_ok=True)
    (root / ".github" / "kernel-maintainers.json").write_text(json.dumps(registry))


def test_maintainers_ok(tmp_path):
    add_kernel(tmp_path, "alpha")
    write_registry(tmp_path, {"alpha": ["octocat", "hubot"]})
    assert kernel_maintainers.check(tmp_path) == []


def test_maintainers_problems(tmp_path):
    add_kernel(tmp_path, "alpha")
    add_kernel(tmp_path, "beta")
    add_kernel(tmp_path, "gamma")
    write_registry(tmp_path, {"alpha": [], "beta": ["@octocat"], "gone": ["x"]})
    problems = "\n".join(kernel_maintainers.check(tmp_path))
    assert "alpha: needs a list of at least one GitHub handle" in problems
    assert "beta: not GitHub handles" in problems
    assert "gamma: missing from" in problems
    assert "gone: in .github/kernel-maintainers.json but there is no kernel" in problems


def test_publish_target_defaults(tmp_path):
    add_kernel(tmp_path, "alpha", version=1)
    t = publish_target.target(tmp_path, "alpha", "rocm-kernels-test")
    assert t["repo-id"] == "rocm-kernels-test/alpha"
    assert t["source-repo-id"] == "rocm-community/alpha"
    assert (t["branch"], t["also-main"], t["load-args"]) == (
        "v1",
        "true",
        "--version 1",
    )
    assert t["config-dir"] == "alpha"


def test_publish_target_branch(tmp_path):
    add_kernel(tmp_path, "alpha", version=2)
    t = publish_target.target(tmp_path, "alpha", "rocm-kernels-test", "staging")
    assert (t["branch"], t["also-main"], t["load-args"]) == (
        "staging",
        "false",
        "--revision staging",
    )


@pytest.mark.parametrize(
    "kernel,org,branch",
    [
        ("alpha", "amd", ""),  # reserved for AMD's packages
        ("alpha", "", ""),  # HUB_ORG not set
        ("alpha", "test-org", "bad branch"),
        ("../alpha", "test-org", ""),
    ],
)
def test_publish_target_rejects(tmp_path, kernel, org, branch):
    add_kernel(tmp_path, "alpha")
    with pytest.raises(ValueError):
        publish_target.target(tmp_path, kernel, org, branch)
