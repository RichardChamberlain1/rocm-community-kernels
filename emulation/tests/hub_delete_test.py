"""emulation/hub_delete.py with a stubbed Hub API.

It shows a repository first, and deletes only after its full id is typed.
"""

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

EMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMULATION))

import hub_delete  # noqa: E402

POC = "rocm-community-poc"
REPO = f"{POC}/scale-triton"
# Built at run time, so that no token-shaped string is in the repository.
FAKE_TOKEN = "hf_" + "NotARealToken" * 3


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def hub(monkeypatch):
    """A stub Hub: answers GETs for REPO, records every request."""
    sent = []
    state = {"exists": True}

    def urlopen(req, timeout=None):
        sent.append(req)
        url = req.full_url
        if req.get_method() == "GET" and not state["exists"]:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if url.endswith("/refs"):
            body = {"branches": [{"name": "v1"}, {"name": "main"}]}
        elif req.get_method() == "GET":
            body = {"id": REPO, "private": True, "lastModified": "2026-11-18T12:00:00Z"}
        else:
            body = {}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    return sent, state


def deletes(sent):
    return [r for r in sent if r.get_method() == "DELETE"]


def test_without_delete_it_only_shows_the_repository(hub, capsys):
    sent, _ = hub
    assert hub_delete.main(["--hub-org", POC, "scale-triton"]) == 0
    out = capsys.readouterr().out
    assert f"https://huggingface.co/kernels/{REPO}" in out
    assert "private:       True" in out
    assert "main, v1" in out
    assert "Nothing deleted" in out
    assert sent and not deletes(sent)
    assert all(r.get_header("Authorization") == f"Bearer {FAKE_TOKEN}" for r in sent)
    assert FAKE_TOKEN not in out


def test_deletes_only_after_typing_the_full_id(hub, monkeypatch, capsys):
    sent, _ = hub
    monkeypatch.setattr("builtins.input", lambda prompt: REPO)
    assert hub_delete.main(["--hub-org", POC, "scale-triton", "--delete"]) == 0
    (req,) = deletes(sent)
    assert req.full_url == "https://huggingface.co/api/repos/delete"
    assert json.loads(req.data) == {
        "type": "kernel",
        "organization": POC,
        "name": "scale-triton",
    }
    assert "deleted" in capsys.readouterr().out


@pytest.mark.parametrize("typed", ["", "y", "yes", "scale-triton", f"{REPO}x"])
def test_anything_else_keeps_it(hub, monkeypatch, typed):
    sent, _ = hub
    monkeypatch.setattr("builtins.input", lambda prompt: typed)
    assert hub_delete.main(["--hub-org", POC, "scale-triton", "--delete"]) == 0
    assert not deletes(sent)


def test_no_terminal_keeps_it(hub, monkeypatch):
    sent, _ = hub

    def no_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_input)
    hub_delete.main(["--hub-org", POC, "scale-triton", "--delete"])
    assert not deletes(sent)


def test_missing_repository_is_reported_not_deleted(hub, monkeypatch, capsys):
    sent, state = hub
    state["exists"] = False
    monkeypatch.setattr("builtins.input", lambda prompt: REPO)
    assert hub_delete.main(["--hub-org", POC, "scale-triton", "--delete"]) == 0
    assert "not found" in capsys.readouterr().out
    assert not deletes(sent)


@pytest.mark.parametrize(
    "args",
    [
        ["--hub-org", "rocm-community", "scale-triton"],
        ["--hub-org", "amd", "scale-triton"],
        ["--hub-org", POC, "aiter-rope"],  # reserved for AMD
    ],
)
def test_refuses_other_orgs_and_reserved_names(hub, args):
    sent, _ = hub
    with pytest.raises(SystemExit):
        hub_delete.main([*args, "--delete"])
    assert not sent


def test_needs_a_token(hub, monkeypatch):
    sent, _ = hub
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        hub_delete.main(["--hub-org", POC, "scale-triton"])
    assert not sent
