from pathlib import Path

import yaml

from smolvla_analysis import runtime


def test_libero_runtime_config_relocates_package_and_assets(tmp_path, monkeypatch):
    project = tmp_path / "project"
    package_site = tmp_path / "site-packages"
    package_root = package_site / "libero/libero"
    (package_root / "bddl_files").mkdir(parents=True)
    (package_root / "init_files").mkdir()
    (project / "archive/full_experiment/checkpoints/libero_datasets").mkdir(parents=True)
    (project / "archive/full_experiment/checkpoints/libero_assets").mkdir(parents=True)
    monkeypatch.setattr(runtime.sys, "path", [str(package_site)])

    destination = runtime._prepare_libero_runtime_config(project, tmp_path / "resolved")
    config = yaml.safe_load((destination / "config.yaml").read_text())

    assert Path(config["benchmark_root"]) == package_root
    assert Path(config["bddl_files"]) == package_root / "bddl_files"
    assert Path(config["init_states"]) == package_root / "init_files"
    assert Path(config["assets"]).name == "libero_assets"
