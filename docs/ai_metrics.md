# AeroVoxel AI modules and evidence

This is the side-by-side metrics snapshot for the project's AI modules.

| Module | Input → output | Evidence | Validation status |
|---|---|---|---|
| Educational surrogate (baseline AI module) | 2D mask → `cd_force_proxy` | [`backend/app/ml/artifacts/last_eval.json`](../backend/app/ml/artifacts/last_eval.json), trained joblib artifact, ablation scripts | CPU metrics committed; labels are this project's LBM proxies |
| Stable Fast 3D neural reconstruction | phone image → mesh → center-slice mask | [`backend/app/recon/README.md`](../backend/app/recon/README.md), `scripts/run_recon_demo.py`, per-run `result.json` | Requires optional CUDA/SF3D environment; no local GPU verification in this checkout |

## Recon evaluation record

The table below is deliberately a record, not fabricated benchmark data. Populate one row per real, consented capture after running `scripts/run_recon_demo.py`; the runner records wall time, device, peak VRAM, mesh output, and downstream mask statistics.

| Case ID | Input type | Recon success? | Notes | Downstream sim ran? | Wall time | Peak VRAM |
|---|---|---:|---|---:|---:|---:|
| R1 | Clean phone capture | BLOCKED | No CUDA/SF3D runtime available in current environment | N | — | — |
| R2 | Clean phone capture | NOT RUN | Awaiting consented capture and GPU run | N | — | — |
| R3 | Clean phone capture | NOT RUN | Awaiting consented capture and GPU run | N | — | — |
| R4 | Clean phone capture | NOT RUN | Awaiting consented capture and GPU run | N | — | — |
| R5 | Clean phone capture | NOT RUN | Awaiting consented capture and GPU run | N | — | — |

The current machine reports CPU-only PyTorch and `nvidia-smi` permission failure. Therefore GPU success, five real captures, and GPU performance are not claimed here.
