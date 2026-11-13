"""The real `kernels` loader against the emulated Hub (no GPU or network needed).

Builds a tiny Python-only kernel variant with a real digest, publishes it with
local_hub, and loads it through ci_verify_load.py with HF_HUB_OFFLINE=1, the
way the CI's verify stage does. Skipped where torch or kernels isn't installed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

kernels = pytest.importorskip("kernels")
pytest.importorskip("torch")

EMULATION = Path(__file__).resolve().parents[1]
SCRIPTS = EMULATION.parent / "scripts"
sys.path.insert(0, str(EMULATION))

import local_hub  # noqa: E402

REPO = "test-org/demo"
KERNEL_TEST = """\
import kernels
import pytest
import torch

demo = kernels.get_kernel("test-org/demo", version=1)


@pytest.mark.kernels_ci
def test_double():
    assert torch.equal(demo.double(torch.tensor([1.0, 2.0])), torch.tensor([2.0, 4.0]))
"""


def make_build(root: Path, factor: int = 2) -> Path:
    from kernels._rust import Digest, DigestAlgorithm
    from kernels.backends import _backend

    backend = _backend().name
    variant = (
        root / "build" / f"torch-{backend}"
    )  # Python-only (noarch) variant for this machine
    variant.mkdir(parents=True)
    (variant / "__init__.py").write_text(f"def double(x):\n    return x * {factor}\n")
    digest = Digest.hash_variant(DigestAlgorithm.SHA256, variant)
    metadata = {
        "name": "demo",
        "id": "demo_test",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": backend},
        "digest": {"algorithm": "sha256", "files": digest.files},
    }
    (variant / "metadata.json").write_text(json.dumps(metadata))
    return root / "build"


def load(tmp_path: Path, hub: Path, *args: str) -> tuple[int, dict]:
    out = tmp_path / f"report-{len(list(tmp_path.glob('report-*')))}.json"
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "LOCAL_KERNELS")
    }
    env.update(
        HF_HUB_OFFLINE="1", KERNELS_CACHE=str(hub), HF_HOME=str(tmp_path / "hf-home")
    )
    cmd = [
        sys.executable,
        str(SCRIPTS / "ci_verify_load.py"),
        "--repo-id",
        REPO,
        "--expect-under",
        str(local_hub.repo_folder(hub, REPO)),
        "--json",
        str(out),
        *args,
    ]
    rc = subprocess.run(
        cmd, env=env, cwd=tmp_path, capture_output=True, text=True
    ).returncode
    return rc, json.loads(out.read_text())


def test_get_kernel_from_emulated_hub(tmp_path):
    hub = tmp_path / "hub"
    pushed = local_hub.push(hub, REPO, make_build(tmp_path / "v1"), "v1")
    rc, report = load(
        tmp_path, hub, "--version", "1", "--smoke", "k.double(torch.tensor([1.0, 2.0]))"
    )
    assert rc == 0, report
    assert report["ok"] and report["commit"] == pushed.commit
    assert report["variant"] == pushed.variants[0]
    assert report["digest_warnings"] == []
    assert report["smoke"]["shape"] == [2]


def test_staging_branch_is_separate_from_version(tmp_path):
    hub = tmp_path / "hub"
    v1 = local_hub.push(hub, REPO, make_build(tmp_path / "v1"), "v1")
    staging = local_hub.push(
        hub, REPO, make_build(tmp_path / "staging", factor=3), "staging"
    )
    rc, rep = load(
        tmp_path,
        hub,
        "--revision",
        "staging",
        "--smoke",
        "k.double(torch.tensor([1.0]))",
    )
    assert rc == 0 and rep["commit"] == staging.commit
    rc, rep = load(
        tmp_path, hub, "--version", "1", "--smoke", "k.double(torch.tensor([1.0]))"
    )
    assert rc == 0 and rep["commit"] == v1.commit


def test_tampered_files_fail_the_digest_check(tmp_path):
    hub = tmp_path / "hub"
    pushed = local_hub.push(hub, REPO, make_build(tmp_path / "v1"), "v1")
    init = next(Path(pushed.snapshot, "build").glob("*/__init__.py"))
    init.write_text("def double(x):\n    return x\n")
    rc, report = load(tmp_path, hub, "--version", "1")
    assert rc == 1 and not report["ok"]
    assert "digest" in report["error"].lower() or "match" in report["error"].lower()


def test_kernel_tests_run_against_emulated_hub(tmp_path):
    hub = tmp_path / "hub"
    local_hub.push(hub, REPO, make_build(tmp_path / "v1"), "v1")
    tests = tmp_path / "kernel" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_demo.py").write_text(KERNEL_TEST)
    (tests / "conftest.py").write_text(
        "def pytest_configure(config):\n    config.addinivalue_line('markers', 'kernels_ci: CI subset')\n"
    )
    rc, report = load(tmp_path, hub, "--version", "1", "--tests", str(tests))
    assert rc == 0, report
    assert report["tests_exit"] == 0
