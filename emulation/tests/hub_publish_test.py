"""emulation/hub_publish.py and `ci_emulate.py --hub-org` without network or token.

The Hub API is stubbed; the dry runs execute nothing. They check that uploads are
private, only go to rocm-community-poc, and that no token is read or printed.
"""

import io
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

EMULATION = Path(__file__).resolve().parents[1]
FIXTURES = EMULATION / "kernels"
LIST = EMULATION / "test-kernels.toml"
sys.path.insert(0, str(EMULATION))

import hub_publish  # noqa: E402

POC = "rocm-community-poc"
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN")
# Built at run time so that no token-shaped string is in the repository.
FAKE_TOKEN = "hf_" + "NotARealToken" * 3


def test_upload_cmd_is_private(tmp_path):
    cmd = [str(c) for c in hub_publish.upload_cmd(tmp_path, f"{POC}/x", "v1", tmp_path)]
    assert cmd[cmd.index("--repo-id") + 1] == f"{POC}/x"
    assert cmd[cmd.index("--branch") + 1] == "v1"
    assert "--private" in cmd


@pytest.mark.parametrize(
    "repo_id",
    [
        "rocm-community/x",
        "amd/x",
        "someone/x",
        f"{POC}/aiter-rope",  # reserved for AMD
        f"{POC}/",
        "x",
    ],
)
def test_refuses_other_orgs_and_reserved_names(tmp_path, repo_id):
    with pytest.raises(ValueError):
        hub_publish.upload_cmd(tmp_path, repo_id, "v1", tmp_path / "out.json")


def test_target_swaps_the_org_and_keeps_the_name():
    t = hub_publish.target(FIXTURES / "scale-triton", POC)
    assert t["repo-id"] == f"{POC}/scale-triton"
    assert t["source-repo-id"] == "rocm-community/scale-triton"
    assert (t["branch"], t["also-main"]) == ("v1", "true")
    with pytest.raises(ValueError):
        hub_publish.target(FIXTURES / "scale-triton", "rocm-community")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def requests(monkeypatch):
    """Record the requests hub_publish sends; answer with `answer`."""
    sent = []
    answer = {"body": b"{}"}

    def urlopen(req, timeout=None):
        sent.append(req)
        return FakeResponse(answer["body"])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return sent, answer


@pytest.mark.parametrize("private", [True, False])
def test_is_private(requests, private):
    sent, answer = requests
    answer["body"] = json.dumps({"id": f"{POC}/x", "private": private}).encode()
    assert hub_publish.is_private(f"{POC}/x", FAKE_TOKEN) is private
    (req,) = sent
    assert req.get_method() == "GET"
    assert req.full_url == f"https://huggingface.co/api/kernels/{POC}/x"
    assert req.get_header("Authorization") == f"Bearer {FAKE_TOKEN}"


def emulate(tmp_path, *args, token=None):
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS}
    if token:
        env["HF_TOKEN"] = token
    cmd = [sys.executable, str(EMULATION / "ci_emulate.py"), *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=tmp_path)


def dry_run(tmp_path, *extra, token=None):
    return emulate(
        tmp_path,
        "run",
        "--approved",
        str(LIST),
        "--hub-org",
        POC,
        "--dry-run",
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "t",
        *extra,
        token=token,
    )


def test_dry_run_needs_no_token_and_uploads_privately(tmp_path):
    r = dry_run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    uploads = [line for line in r.stdout.splitlines() if " upload " in line]
    assert len(uploads) == 4  # scale-triton and scale-hip, each to v1 and main
    assert all("--private" in line for line in uploads)
    for name in ("scale-triton", "scale-hip"):
        assert sum(f"{POC}/{name} " in line for line in uploads) == 2
    assert not any("bad-community" in line for line in uploads)
    summary = json.loads((tmp_path / "out" / "t" / "summary.json").read_text())
    stages = {k["kernel"]: k["stages"] for k in summary["kernels"]}
    assert stages["bad-community"]["publish"]["status"] == "skip"
    report = (tmp_path / "out" / "t" / "summary.md").read_text()
    assert "nothing was sent" in report
    assert "Published privately" not in report


def test_dry_run_never_shows_the_token(tmp_path):
    r = dry_run(tmp_path, token=FAKE_TOKEN)
    assert r.returncode == 0, r.stdout + r.stderr
    assert FAKE_TOKEN not in r.stdout + r.stderr
    for path in (tmp_path / "out").rglob("*"):
        if path.is_file():
            assert FAKE_TOKEN not in path.read_text(errors="ignore"), path


def test_publish_note_names_a_skip_instead_of_a_publish():
    import ci_emulate
    from kernel_ci import KernelRun, StageResult

    class Ctx:
        hub_org = POC
        dry_run = False

    kr = KernelRun({}, Path("scale-triton"), Path("out"))
    kr.results["publish"] = StageResult("skip", message="no build to publish")
    note = ci_emulate.publish_note(
        Ctx(), [kr], ["validate", "build", "test", "publish", "verify", "benchmark"]
    )
    assert note == "Nothing published: scale-triton (no build to publish)."


def test_real_run_without_token_stops_before_any_stage(tmp_path):
    r = emulate(
        tmp_path,
        "run",
        "--approved",
        str(LIST),
        "--hub-org",
        POC,
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "t",
    )
    assert r.returncode != 0
    assert "HF_TOKEN" in r.stderr
    assert not (tmp_path / "out" / "t").exists()


@pytest.mark.parametrize("org", ["rocm-community", "amd", "someone"])
def test_other_orgs_are_refused(tmp_path, org):
    r = emulate(tmp_path, "run", "--approved", str(LIST), "--hub-org", org, "--dry-run")
    assert r.returncode != 0
    assert "refusing" in r.stderr
