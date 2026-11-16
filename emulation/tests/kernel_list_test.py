"""The emulation's test kernels (emulation/test-kernels.toml) against the rule checks.

The kernels live in emulation/kernels; TEST_KERNELS_ROOT points elsewhere instead.
Community-rule entries are checked with ci/community.toml, so these tests show the
community rules pass the fixtures and stop the deliberately broken one.
"""

import json
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
ROOT = Path(os.environ.get("TEST_KERNELS_ROOT") or LIST.parent / DOC["kernels_root"])
CASES = DOC["kernel"]

pytestmark = pytest.mark.skipif(
    not all((ROOT / c["path"]).is_dir() for c in CASES),
    reason=f"the test kernels aren't all in {ROOT}",
)


def community(case) -> bool:
    return case.get("rules", "community") == "community"


def test_list_excludes_composable_kernel_and_bundle():
    assert all(
        "-ck" not in case["path"] and case["path"] != "aiter-kernels" for case in CASES
    )


def test_list_covers_both_rule_sets():
    assert any(community(c) for c in CASES) and not all(community(c) for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["path"] for c in CASES])
def test_wrapping_matches_expectation(case):
    info, findings = check_wrapping.check_kernel(
        ROOT / case["path"], community=community(case)
    )
    errors = [f for f in findings if f.level == "error"]
    expected = case.get("expect", {}).get("validate", "pass")
    assert (not errors) == (expected == "pass"), [f.message for f in errors]
    assert info.version is not None
    if community(case) and expected == "pass":
        assert info.repo_id == f"rocm-community/{case['path']}"
    elif not community(case):
        assert info.repo_id.startswith("amd/")


def test_bad_community_fails_for_the_planted_reasons():
    _, findings = check_wrapping.check_kernel(ROOT / "bad-community", community=True)
    errors = {f.code for f in findings if f.level == "error"}
    assert errors == {"reserved-org", "card-notice", "tests-kernels-ci"}


def test_amd_kernels_fail_the_community_rules():
    """The aiter kernels use names and orgs the community rules reserve for AMD."""
    for case in CASES:
        if not community(case):
            _, findings = check_wrapping.check_kernel(
                ROOT / case["path"], community=True
            )
            errors = {f.code for f in findings if f.level == "error"}
            assert {"reserved-name", "reserved-org"} <= errors, case["path"]


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
    # The expected validate failures don't fail the run.
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.loads((tmp_path / "t" / "summary.json").read_text())
    stages = {k["kernel"]: k["stages"] for k in summary["kernels"]}
    assert set(stages) == {case["path"] for case in CASES}
    # The gate stops bad-community: validate fails and nothing after it runs.
    bad = stages["bad-community"]
    assert bad["validate"]["status"] == "fail"
    assert all(
        s["status"] == "skip" for name, s in bad.items() if name != "validate"
    ), bad
