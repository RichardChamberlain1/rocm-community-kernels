"""The AMD test cases (emulation/test-kernels.toml) against their sources.

The kernels are in the repository root; TEST_KERNELS_ROOT points elsewhere instead.
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

EMULATION = Path(__file__).resolve().parents[1]
LIST = EMULATION / "test-kernels.toml"
sys.path.insert(0, str(EMULATION.parent / "scripts"))

import check_wrapping  # noqa: E402

DOC = tomllib.loads(LIST.read_text())
ROOT = Path(os.environ.get("TEST_KERNELS_ROOT") or EMULATION.parent)
CASES = DOC["kernel"]

pytestmark = pytest.mark.skipif(
    not all((ROOT / c["path"]).is_dir() for c in CASES),
    reason=f"the test kernels aren't all in {ROOT}",
)


def test_list_excludes_composable_kernel_and_bundle():
    assert all(
        "-ck" not in case["path"] and case["path"] != "aiter-kernels" for case in CASES
    )


@pytest.mark.parametrize("case", CASES, ids=[c["path"] for c in CASES])
def test_wrapping_matches_expectation(case):
    info, findings = check_wrapping.check_kernel(
        ROOT / case["path"], community=case.get("rules") == "community"
    )
    errors = [f for f in findings if f.level == "error"]
    expected = case.get("expect", {}).get("validate", "pass")
    assert (not errors) == (expected == "pass"), [f.message for f in errors]
    assert info.repo_id.startswith("amd/") and info.version is not None


def test_dry_run_of_the_list(tmp_path):
    cmd = [
        sys.executable,
        str(EMULATION / "ci_emulate.py"),
        "run",
        "--approved",
        str(LIST),
        "--kernels-root",
        str(ROOT),
        "--dry-run",
        "--out",
        str(tmp_path),
        "--run-id",
        "t",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (
        r.stdout + r.stderr
    )  # the expected validate failure doesn't fail the run
    summary = (tmp_path / "t" / "summary.md").read_text()
    for case in CASES:
        assert f"| {case['path']} |" in summary
    assert "✓ expected" in summary
