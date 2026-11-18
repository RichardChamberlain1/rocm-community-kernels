"""Guard: which files may write to the Hugging Face Hub, and how.

Covers the emulation and the CI scripts it runs:

- uploads: only emulation/hub_publish.py (`ci_emulate.py run --hub-org
  rocm-community-poc`), always with --private, as .github/workflows/publish.yml;
- deletion: only emulation/hub_delete.py, which is run by hand and which nothing
  imports, so no pipeline can delete;
- nothing else writes.
"""

import re
import sys
from pathlib import Path

EMULATION = Path(__file__).resolve().parents[1]
SCRIPTS = EMULATION.parent / "scripts"
sys.path.insert(0, str(EMULATION))

import hub_publish  # noqa: E402

UPLOADER = EMULATION / "hub_publish.py"
DELETER = EMULATION / "hub_delete.py"
# kernel-builder upload / hf upload as a command argument
UPLOAD = [r"[\"']upload[\"']"]
DELETE = [
    r"[\"']DELETE[\"']",
    r"/api/repos/",
    r"\bdelete_(?:repo|branch|file|folder)\b",
]
OTHER_WRITES = [
    r"\bupload_(?:file|folder|large_folder)\b",
    r"\bcreate_(?:commit|repo|branch|tag)\b",
    r"\bHfApi\s*\(",
    r"huggingface-cli",
]


def code_files() -> list[Path]:
    paths = sorted(EMULATION.glob("*.py")) + sorted(SCRIPTS.glob("*.py"))
    assert UPLOADER in paths and DELETER in paths
    return paths


def matches(path: Path, patterns: list[str]) -> list[str]:
    text = path.read_text()
    return [p for p in patterns if re.search(p, text)]


def test_uploads_only_in_hub_publish():
    for path in code_files():
        if path != UPLOADER:
            assert not matches(path, UPLOAD), path.name


def test_deletion_only_in_hub_delete():
    for path in code_files():
        if path != DELETER:
            assert not matches(path, DELETE), path.name
    assert matches(DELETER, DELETE)


def test_no_other_hub_writes():
    for path in code_files():
        assert not matches(path, OTHER_WRITES), path.name


def test_nothing_imports_the_delete_script():
    for path in code_files():
        if path != DELETER:
            imports = r"^\s*(?:import|from)\s+hub_delete\b"
            assert not re.search(imports, path.read_text(), re.M), path.name


def test_hub_publish_only_targets_the_poc_org():
    assert hub_publish.ALLOWED_ORG == "rocm-community-poc"


def test_every_upload_is_private(tmp_path):
    cmd = [
        str(c)
        for c in hub_publish.upload_cmd(
            tmp_path, "rocm-community-poc/scale-triton", "v1", tmp_path / "out.json"
        )
    ]
    assert "upload" in cmd and "--private" in cmd
    # One upload command in the uploader, and it is the one tested here.
    assert UPLOADER.read_text().count('"upload"') == 1
