import importlib.util
from pathlib import Path

from smolvla_analysis.phase3_crd import (
    atomic_write_json,
    iter_branch_specs,
    legacy_cross_instance_branch_ids,
)


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_phase3_crd.py"
    spec = importlib.util.spec_from_file_location("run_phase3_crd_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_branch_refresh_preserves_old_payloads_and_is_resumable(tmp_path):
    runner = _runner_module()
    run_dir = tmp_path / "run"
    branches_dir = run_dir / "branches"
    branches_dir.mkdir(parents=True)
    manifest = {"run_id": "test", "branch_refreshes": []}
    atomic_write_json(run_dir / "manifest.json", manifest)
    targets = legacy_cross_instance_branch_ids()
    for branch_id in targets:
        atomic_write_json(branches_dir / f"{branch_id}.json", {"branch_id": branch_id, "old": True})

    runner._prepare_legacy_branch_refresh(run_dir, manifest, branches_dir)

    entry = manifest["branch_refreshes"][0]
    assert entry["status"] == "running"
    assert len(entry["old_sha256"]) == 30
    assert not any((branches_dir / f"{branch_id}.json").exists() for branch_id in targets)
    backup_dir = run_dir / "superseded_branches" / entry["kind"]
    assert all((backup_dir / f"{branch_id}.json").is_file() for branch_id in targets)

    # Simulate an interrupted refresh being completed by a later normal resume.
    for branch_id in targets:
        atomic_write_json(branches_dir / f"{branch_id}.json", {"branch_id": branch_id, "old": False})
    runner._prepare_legacy_branch_refresh(run_dir, manifest, branches_dir)
    runner._finalize_branch_refreshes(run_dir, manifest, branches_dir)

    assert entry["status"] == "complete"
    assert len(entry["new_sha256"]) == 30
    assert set(entry["old_sha256"]) == set(entry["new_sha256"])


def test_branch_replay_comparison_ignores_only_runtime_provenance_and_timing():
    runner = _runner_module()
    existing = {"success": True, "wall_time_s": 1.0}
    replayed = {
        "success": True,
        "wall_time_s": 2.0,
        "source_reconstruction": {"mode": "archive_action_replay_current_process"},
    }
    assert runner._compare_branch_payloads(existing, replayed)["semantic_exact_match"]
    replayed["success"] = False
    comparison = runner._compare_branch_payloads(existing, replayed)
    assert not comparison["semantic_exact_match"]
    assert comparison["changed_top_level_fields"] == ["success"]


def test_uncertified_refresh_selects_exactly_payloads_without_source_provenance(tmp_path):
    runner = _runner_module()
    run_dir = tmp_path / "run"
    branches_dir = run_dir / "branches"
    branches_dir.mkdir(parents=True)
    manifest = {"run_id": "test", "branch_refreshes": []}
    atomic_write_json(run_dir / "manifest.json", manifest)
    branches = iter_branch_specs()
    for index, branch in enumerate(branches):
        payload = {"branch_id": branch.branch_id}
        if index >= 84:
            payload["source_reconstruction"] = {"mode": "archive_action_replay_current_process"}
        atomic_write_json(branches_dir / f"{branch.branch_id}.json", payload)

    runner._prepare_uncertified_branch_refresh(run_dir, manifest, branches_dir)

    entry = manifest["branch_refreshes"][0]
    assert entry["status"] == "running"
    assert len(entry["target_branch_ids"]) == 84
    assert len(list(branches_dir.glob("*.json"))) == 76
    assert len(
        list((run_dir / "superseded_branches/legacy_uncertified_branch_roots").glob("*.json"))
    ) == 84
