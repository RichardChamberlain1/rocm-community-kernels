import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_emulate  # noqa: E402


def test_kernel_flag_skips_approved_list(tmp_path):
    approved = tmp_path / "approved.toml"
    approved.write_text('[[kernel]]\npath = "listed"\n')
    args = ci_emulate.parse_args(["run", "--kernel", "pr-kernel"])
    assert [e["path"] for e in ci_emulate.load_entries(args)] == ["pr-kernel"]
    args = ci_emulate.parse_args(
        ["run", "--approved", str(approved), "--kernel", "pr-kernel"]
    )
    assert [e["path"] for e in ci_emulate.load_entries(args)] == ["listed", "pr-kernel"]


def test_emulation_adds_publish_stages_to_the_ci_stages():
    assert ci_emulate.STAGES[:3] == ["validate", "build", "test"]
    assert set(ci_emulate.STAGE_FUNCS) == set(ci_emulate.STAGES)
