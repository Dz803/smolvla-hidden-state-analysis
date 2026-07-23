#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REQUIRED_CHECKPOINT_FILES = {
    "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
}


def check(name, function):
    try:
        detail = function()
        return {"name": name, "ok": True, "detail": str(detail)}
    except Exception as error:
        return {"name": name, "ok": False, "detail": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints/smolvla_libero")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/doctor.json")
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "egl"

    def gpu():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        value = torch.ones(1024, device="cuda")
        if float(value.sum()) != 1024:
            raise RuntimeError("CUDA arithmetic check failed")
        return f"{torch.cuda.get_device_name(0)}; torch={torch.__version__}; cuda={torch.version.cuda}"

    def egl():
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><light pos='0 0 2'/><geom type='sphere' size='.1'/></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=32, width=32)
        renderer.update_scene(data)
        pixels = renderer.render()
        renderer.close()
        return f"MuJoCo {mujoco.__version__}; EGL frame={pixels.shape}"

    def packages():
        names = ("lerobot", "hf-libero", "mujoco", "torch", "transformers", "numpy", "zarr", "pyarrow", "pandas", "scikit-learn")
        return {name: importlib.metadata.version(name) for name in names}

    def pip_check():
        result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout.strip() or result.stderr.strip())
        return result.stdout.strip()

    def checkpoint():
        missing = REQUIRED_CHECKPOINT_FILES - {path.name for path in args.checkpoint.glob("*")}
        if missing:
            raise FileNotFoundError(f"missing checkpoint files: {sorted(missing)}")
        return args.checkpoint.resolve()

    checks = [
        check("linux", lambda: sys.platform if sys.platform.startswith("linux") else (_ for _ in ()).throw(RuntimeError(sys.platform))),
        check("disk", lambda: f"free_gib={shutil.disk_usage(PROJECT).free / 2**30:.1f}"),
        check("dependencies", packages),
        check("pip_check", pip_check),
        check("cuda", gpu),
        check("mujoco_egl", egl),
        check("checkpoint", checkpoint),
    ]
    report = {
        "timestamp": datetime.now(UTC).isoformat(), "mujoco_gl": os.environ["MUJOCO_GL"],
        "ok": all(item["ok"] for item in checks), "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for item in checks:
        print(f"{'PASS' if item['ok'] else 'FAIL'} {item['name']}: {item['detail']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

