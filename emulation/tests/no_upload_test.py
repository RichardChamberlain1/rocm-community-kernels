"""Guard: the CI emulation must never push anything to the Hugging Face Hub.

Covers the emulation and the CI scripts it runs. The only real upload is in
.github/workflows/publish.yml, behind an approval.
"""

import re
from pathlib import Path

EMULATION = Path(__file__).resolve().parents[1]
SCRIPTS = EMULATION.parent / "scripts"

FORBIDDEN = [
    r"[\"']upload[\"']",  # kernel-builder upload / hf upload as a command argument
    r"\bupload_(?:file|folder|large_folder)\b",
    r"\bcreate_(?:commit|repo|branch|tag)\b",
    r"\bHfApi\s*\(",
    r"huggingface-cli",
]


def test_no_hub_write_calls():
    paths = sorted(EMULATION.glob("*.py")) + sorted(SCRIPTS.glob("*.py"))
    assert any(p.name == "ci_emulate.py" for p in paths)
    for path in paths:
        text = path.read_text()
        for pattern in FORBIDDEN:
            assert not re.search(pattern, text), f"{path.name} matches {pattern!r}"
