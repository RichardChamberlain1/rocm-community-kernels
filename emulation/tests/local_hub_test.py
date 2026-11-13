import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import local_hub  # noqa: E402


def make_build(
    root: Path, variants=("torch215-rocm100-x86_64-linux", "torch-rocm"), payload="a"
) -> Path:
    build = root / "build"
    for v in variants:
        (build / v).mkdir(parents=True)
        (build / v / "metadata.json").write_text('{"name": "k"}')
        (build / v / "__init__.py").write_text(f"X = {payload!r}\n")
    return build


def test_push_writes_hub_cache_layout(tmp_path):
    build = make_build(tmp_path)
    hub = tmp_path / "hub"
    res = local_hub.push(hub, "rocm-community/demo", build, "v1")
    folder = hub / "kernels--rocm-community--demo"
    assert (folder / "refs" / "v1").read_text() == res.commit
    assert len(res.commit) == 40
    snap = folder / "snapshots" / res.commit
    assert (snap / "build" / "torch-rocm" / "metadata.json").is_file()
    assert res.variants == ["torch-rocm", "torch215-rocm100-x86_64-linux"]
    assert local_hub.resolve(hub, "rocm-community/demo", "v1") == snap
    assert local_hub.resolve(hub, "rocm-community/demo", res.commit) == snap


def test_commit_is_deterministic_and_content_addressed(tmp_path):
    a = make_build(tmp_path / "a")
    b = make_build(tmp_path / "b")
    c = make_build(tmp_path / "c", payload="changed")
    assert local_hub.tree_commit(a) == local_hub.tree_commit(b)
    assert local_hub.tree_commit(a) != local_hub.tree_commit(c)


def test_branches_are_independent(tmp_path):
    hub = tmp_path / "hub"
    r1 = local_hub.push(hub, "org/k", make_build(tmp_path / "one"), "v1")
    r2 = local_hub.push(
        hub, "org/k", make_build(tmp_path / "two", payload="new"), "staging"
    )
    assert local_hub.refs(hub, "org/k") == {"staging": r2.commit, "v1": r1.commit}
    assert r1.commit != r2.commit
    # Re-publishing the same content to v1 is a no-op for the snapshot.
    again = local_hub.push(hub, "org/k", make_build(tmp_path / "three"), "v1")
    assert again.commit == r1.commit


def test_symlinked_files_are_copied_as_content(tmp_path):
    build = make_build(tmp_path)
    target = tmp_path / "blob"
    target.write_text("blob content")
    (build / "torch-rocm" / "linked.txt").symlink_to(target)
    res = local_hub.push(tmp_path / "hub", "org/k", build, "v1")
    copied = Path(res.snapshot) / "build" / "torch-rocm" / "linked.txt"
    assert not copied.is_symlink() and copied.read_text() == "blob content"


@pytest.mark.parametrize("repo_id", ["noslash", "a/b/c", "../x", "org/"])
def test_rejects_bad_repo_ids(tmp_path, repo_id):
    with pytest.raises(local_hub.LocalHubError):
        local_hub.push(tmp_path / "hub", repo_id, make_build(tmp_path), "v1")


def test_rejects_bad_branch_and_empty_build(tmp_path):
    with pytest.raises(local_hub.LocalHubError):
        local_hub.push(tmp_path / "hub", "org/k", make_build(tmp_path), "../escape")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(local_hub.LocalHubError):
        local_hub.push(tmp_path / "hub", "org/k", empty, "v1")


def test_layout_describes_refs(tmp_path):
    hub = tmp_path / "hub"
    local_hub.push(hub, "org/k", make_build(tmp_path, variants=("torch-rocm",)), "v2")
    assert local_hub.layout(hub, "org/k") == {
        "v2": {"snapshot_exists": True, "variants": ["torch-rocm"]}
    }
