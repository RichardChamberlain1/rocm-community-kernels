import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_wrapping  # noqa: E402

BUILD_TOML = """\
[general]
name = "demo-kernel"
version = 1
edition = 6
license = "Apache-2.0"
backends = ["rocm"]

[general.hub]
repo-id = "rocm-community/demo-kernel"

[torch]
src = ["torch-ext/torch_binding.cpp"]

[kernel.demo_rocm]
language = "cpp-rocm"
depends = ["torch"]
rocm-archs = ["gfx950"]
src = ["demo/kernel.hip"]
"""

TEST = """\
import kernels
import pytest

demo = kernels.get_kernel("rocm-community/demo-kernel", version=1)


@pytest.mark.kernels_ci
def test_demo():
    assert demo is not None
"""


def make_kernel(
    root: Path,
    toml=BUILD_TOML,
    init="from ._ops import ops\n",
    binding=None,
    test=TEST,
    name="demo-kernel",
):
    k = root / name
    (k / "torch-ext" / "demo_kernel").mkdir(parents=True)
    (k / "tests").mkdir()
    (k / "flake.nix").write_text("{}\n")
    (k / "build.toml").write_text(toml)
    (k / "README.md").write_text("# demo\n\nThis kernel is not supported by AMD.\n")
    (k / "torch-ext" / "demo_kernel" / "__init__.py").write_text(init)
    (k / "torch-ext" / "torch_binding.cpp").write_text(
        binding
        or "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {}\nREGISTER_EXTENSION(TORCH_EXTENSION_NAME)\n"
    )
    (k / "tests" / "test_demo.py").write_text(test)
    return k


def codes(findings, level="error"):
    return {f.code for f in findings if f.level == level}


def test_good_kernel_passes(tmp_path):
    info, findings = check_wrapping.check_kernel(make_kernel(tmp_path), community=True)
    assert codes(findings) == set()
    assert (
        info.repo_id == "rocm-community/demo-kernel"
        and info.version == 1
        and info.framework == "torch"
    )


def test_absolute_self_import(tmp_path):
    _, f = check_wrapping.check_kernel(
        make_kernel(tmp_path, init="from demo_kernel._ops import ops\n")
    )
    assert "absolute-import" in codes(f)


@pytest.mark.parametrize(
    "binding",
    ["TORCH_LIBRARY(demo, m) {}\n", "STABLE_TORCH_LIBRARY_IMPL(demo, CUDA, m) {}\n"],
)
def test_fixed_cpp_namespace(tmp_path, binding):
    _, f = check_wrapping.check_kernel(make_kernel(tmp_path, binding=binding))
    assert "op-namespace" in codes(f)


def test_fixed_python_namespace(tmp_path):
    init = 'import torch\n@torch.library.custom_op("demo::op", mutates_args=())\ndef op(x): ...\n'
    _, f = check_wrapping.check_kernel(make_kernel(tmp_path, init=init))
    assert "op-namespace" in codes(f)


def test_missing_repo_id_and_bad_version(tmp_path):
    toml = BUILD_TOML.replace('repo-id = "rocm-community/demo-kernel"', "").replace(
        "version = 1", 'version = "1"'
    )
    _, f = check_wrapping.check_kernel(make_kernel(tmp_path, toml=toml))
    assert {"repo-id", "version"} <= codes(f)


def test_compiled_rocm_needs_archs(tmp_path):
    toml = BUILD_TOML.replace('rocm-archs = ["gfx950"]\n', "")
    _, f = check_wrapping.check_kernel(make_kernel(tmp_path, toml=toml))
    assert "rocm-archs" in codes(f)


def test_tests_must_use_get_kernel_and_kernels_ci(tmp_path):
    _, f = check_wrapping.check_kernel(
        make_kernel(tmp_path, test="import demo_kernel\n\ndef test_x():\n    pass\n")
    )
    assert {"tests-get-kernel", "tests-kernels-ci"} <= codes(f)
    assert "tests-direct-import" in codes(f, "warning")


def test_community_rules(tmp_path):
    toml = BUILD_TOML.replace("rocm-community/demo-kernel", "amd/aiter-demo").replace(
        '"demo-kernel"', '"aiter-demo"'
    )
    k = make_kernel(tmp_path, toml=toml, name="aiter-demo")
    (k / "torch-ext" / "demo_kernel").rename(k / "torch-ext" / "aiter_demo")
    _, f = check_wrapping.check_kernel(k, community=True)
    assert {"reserved-name", "reserved-org"} <= codes(f)
    _, f = check_wrapping.check_kernel(k, community=False)
    assert not ({"reserved-name", "reserved-org"} & codes(f))


def test_community_org_and_name(tmp_path):
    k = make_kernel(
        tmp_path,
        toml=BUILD_TOML.replace("rocm-community/demo-kernel", "someone/other-name"),
    )
    _, f = check_wrapping.check_kernel(k, community=True)
    assert {"community-org", "repo-name"} <= codes(f)
    _, f = check_wrapping.check_kernel(k, community=False)
    assert not ({"community-org", "repo-name"} & codes(f))


def test_archs_must_include_a_ci_gpu(tmp_path):
    toml = BUILD_TOML.replace('["gfx950"]', '["gfx1201"]')
    _, f = check_wrapping.check_kernel(make_kernel(tmp_path, toml=toml), community=True)
    assert "ci-arch" in codes(f)


def test_card_needs_notice(tmp_path):
    k = make_kernel(tmp_path)
    (k / "CARD.md").write_text("# demo\n")  # CARD.md takes precedence over README.md
    _, f = check_wrapping.check_kernel(k, community=True)
    assert "card-notice" in codes(f)
    assert "license-file" in codes(f, "warning")


def test_requires_rocm_backend(tmp_path):
    _, f = check_wrapping.check_kernel(
        make_kernel(tmp_path, toml=BUILD_TOML.replace('["rocm"]', '["cuda"]'))
    )
    assert "backends" in codes(f)
