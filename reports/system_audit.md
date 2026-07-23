# SmolVLA-LIBERO System Audit

Audit date: 2026-07-21; setup verification updated 2026-07-22 (Asia/Shanghai)

## Host

- Hostname: `k8s-master02`
- OS: Ubuntu 22.04.5 LTS
- Kernel: Linux 6.8.0-124-generic, x86_64
- CPU: AMD Ryzen Threadripper 7970X, 32 cores / 64 threads
- RAM: 125 GiB total, approximately 62 GiB available during audit; no swap
- SSD: 3.6 TiB filesystem, approximately 379 GiB available after setup and pilot collection
- Docker: 29.1.3

## GPU

- GPU: NVIDIA GeForce RTX 5090
- Driver: 570.133.07
- VRAM: 32,607 MiB reported by `nvidia-smi`
- Audit utilization: 0%; approximately 533 MiB allocated
- PyTorch-visible compute capability: 12.0

## Existing Python Environment (Discovery Only)

- Conda environment: `lingbotvla`
- Python: 3.12.13
- PyTorch: 2.8.0+cu128
- CUDA build: 12.8
- CUDA available: yes, one device
- LeRobot: 0.4.2
- Transformers: 4.57.3
- NumPy: 1.26.4
- MuJoCo: missing
- LIBERO: missing

This environment is not approved for the experiment. `pip check` reports incompatible LeRobot constraints (Torch, TorchVision, TorchCodec, Datasets) plus unrelated LingBot conflicts. A dedicated environment is required.

## Isolated Experiment Environment

- Location: `.conda-envs/smolvla-libero`
- Python: 3.12.13
- LeRobot: 0.4.4, editable official checkout at commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`
- hf-libero: 0.1.4
- PyTorch: 2.8.0+cu128; CUDA build 12.8
- Transformers: 4.57.3
- NumPy: 2.2.6
- MuJoCo: 3.8.1
- Zarr / PyArrow / pandas: 2.18.7 / 21.0.0 / 2.3.1
- scikit-learn: 1.9.0
- Dependency check: no broken requirements

## Repository

- Repository: `/home/zhongzhengyang/lingbot-vla-v2`
- Branch: `main`
- Commit: `894c40a0080020fdeb953034c93659d96f51080f`
- Status: dirty, with existing modified and untracked LingBot/RoboTwin/UTARS work
- New analysis work is isolated under `experiments/smolvla_libero_failure_analysis/`.
- No repository-local `AGENTS.md` was found; the user-provided professional-English instructions remain applicable.

## Hugging Face and Model Access

- Hugging Face access: public checkpoint metadata and files accessible through the configured proxy; no token was printed or stored
- Checkpoint: `lerobot/smolvla_libero`
- Pinned checkpoint revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
- Access: public and ungated
- Existing SmolVLA/LIBERO cache before setup: none
- Checkpoint and processor assets downloaded successfully
- `model.safetensors`: 906,712,520 bytes; SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`
- The weight hash exactly matches the Hugging Face repository's published LFS object ID
- LIBERO assets are pinned to repository revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`

## Headless Rendering

LIBERO requires Linux, which this host provides. All simulator commands must set:

```bash
export MUJOCO_GL=egl
```

Real MuJoCo EGL rendering and a complete LIBERO rollout were validated on the RTX 5090. Gates A through E passed.
