import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kernel_ci  # noqa: E402
from check_wrapping import KernelInfo  # noqa: E402


def kernel(archs, expect=None, results=None):
    info = KernelInfo("k", "/k", "/k", "org/k", 1, "torch", ["rocm"], "k", archs)
    kr = kernel_ci.KernelRun(
        {"path": "k", "expect": expect or {}}, Path("/k"), Path("/out"), info
    )
    kr.results = results or {}
    return kr


def ctx(gpu_archs, dry_run=False):
    return SimpleNamespace(gpu_archs=gpu_archs, dry_run=dry_run)


def test_arch_gating():
    assert kernel_ci.arch_mismatch(ctx(["gfx950"]), kernel(["gfx1201"])) is not None
    assert (
        kernel_ci.arch_mismatch(ctx(["gfx950"]), kernel(["gfx942", "gfx950"])) is None
    )
    # no archs: may load anywhere
    assert kernel_ci.arch_mismatch(ctx(["gfx950"]), kernel([])) is None
    # GPU unknown: try
    assert kernel_ci.arch_mismatch(ctx([]), kernel(["gfx1201"])) is None
    assert (
        kernel_ci.arch_mismatch(ctx(["gfx950"], dry_run=True), kernel(["gfx1201"]))
        is None
    )


def test_fail_on_skip(tmp_path, monkeypatch):
    (tmp_path / "k").mkdir()

    def validate(ctx, kr):
        kr.info = kernel([]).info
        return kernel_ci.StageResult("pass")

    monkeypatch.setitem(kernel_ci.STAGE_FUNCS, "validate", validate)
    monkeypatch.setitem(
        kernel_ci.STAGE_FUNCS,
        "test",
        lambda ctx, kr: kernel_ci.StageResult("skip", message="no GPU"),
    )
    base = ["run", "--kernels-root", str(tmp_path), "--kernel", "k"]
    base += ["--stages", "validate,test", "--out", str(tmp_path / "out")]
    assert kernel_ci.main([*base, "--run-id", "lenient"]) == 0
    assert kernel_ci.main([*base, "--run-id", "strict", "--fail-on-skip"]) == 1
    summary = (tmp_path / "out" / "strict" / "summary.md").read_text()
    assert "skipped, not allowed here: no GPU" in summary


def test_concurrency_args(tmp_path, monkeypatch):
    limits = tmp_path / "build-concurrency.json"
    limits.write_text('{"big": {"rocm": {"max-jobs": 1, "cores": 8}}}')
    monkeypatch.setattr(kernel_ci, "BUILD_CONCURRENCY", limits)
    assert kernel_ci.concurrency_args("big", "rocm") == [
        "--max-jobs",
        "1",
        "--cores",
        "8",
    ]
    assert kernel_ci.concurrency_args("small", "rocm") == []


def test_expectations():
    R = kernel_ci.StageResult
    ok = kernel([], {"validate": "fail"}, {"validate": R("fail"), "build": R("pass")})
    assert kernel_ci.unexpected(ok) == []
    wrong = kernel([], {"validate": "fail"}, {"validate": R("pass")})
    assert kernel_ci.unexpected(wrong) == ["validate: pass, expected fail"]
    plain = kernel([], {}, {"build": R("fail"), "test": R("skip")})
    assert kernel_ci.unexpected(plain) == ["build: fail"]


def test_commands_never_get_a_token(monkeypatch):
    """Pull request code runs on the GPU runner: no command may see a Hub token."""
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "secret")
    monkeypatch.setenv("LOCAL_KERNELS", "x=y")
    env = kernel_ci.env_for(True, KERNELS_CACHE="/tmp/hub")
    assert (
        "HF_TOKEN" not in env
        and "HUGGING_FACE_HUB_TOKEN" not in env
        and "LOCAL_KERNELS" not in env
    )
    assert env["HF_HUB_OFFLINE"] == "1" and env["KERNELS_CACHE"] == "/tmp/hub"
    assert "HF_HUB_OFFLINE" not in kernel_ci.env_for(False)
