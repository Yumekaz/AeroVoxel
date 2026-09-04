# AeroVoxel AI modules and evidence

This is the side-by-side metrics snapshot for the project's AI modules.

| Module | Input → output | Evidence | Validation status |
|---|---|---|---|
| Educational surrogate (baseline AI module) | 2D mask → `cd_force_proxy` | [`backend/app/ml/artifacts/last_eval.json`](../backend/app/ml/artifacts/last_eval.json), trained joblib artifact, ablation scripts | CPU metrics committed; labels are this project's LBM proxies |
| Depth Anything V2 Small alternative | phone image → relative depth → thickened mesh → center-slice mask | [`backend/app/recon/README.md`](../backend/app/recon/README.md), `scripts/run_recon_demo.py`, `scripts/evaluate_recon.py`, per-run `result.json` | Real CPU-capable ML path; quality is relative-depth single-view geometry |
| Stable Fast 3D primary | phone image → mesh → center-slice mask | [`backend/app/recon/README.md`](../backend/app/recon/README.md), `scripts/run_recon_demo.py`, per-run `result.json` | Preferred when runtime/device/VRAM are available |

## Recon evaluation record

The table below separates the verified local substitute from the still-missing
real phone corpus. The runner records wall time, device, model, serialized mesh
validity, and downstream mask statistics; no ground-truth quality score is
claimed without corresponding geometry.

| Case ID | Input type | Recon success? | Notes | Downstream sim ran? | Wall time | Peak VRAM |
|---|---|---:|---|---:|---:|---:|
| SYNTH_01–05 | Synthetic phone-like fixtures | 5/5 SUCCEEDED | Depth Anything V2 Small; CPU; watertight GLB and non-empty `(64,128)` mask for every case | Not run | 46.448 s total; 3.684–20.218 s per case in the verified final run | N/A (CPU) |
| R1–R5 | Real consented phone captures | NOT RUN | No real phone corpus is present in the repository/environment | N | — | — |

For the verified final run, the first case loaded the model in 2.663 s and
inference ranged from 0.762–1.136 s across the five CPU cases. All five
serialized meshes reported finite vertices, zero degenerate faces, and
`watertight=true`. These measurements describe this Windows laptop and are
not transferable performance guarantees.

The current machine exposes an RTX 3050 Laptop GPU through Windows inventory, but the device is in Code 43 (`CM_PROB_FAILED_POST_START`), `nvidia-smi` cannot access it, and the installed PyTorch is CPU-only. The real CPU alternative is verified; GPU/SF3D evidence remains blocked by the host driver and the device's 4 GB VRAM versus SF3D's documented roughly 6 GB default. A CUDA 12.6 requirements file is provided for later activation. Run `scripts/check_recon_environment.py` after driver repair; run `scripts/evaluate_recon.py` with five consented `PHONE_CAPTURE` rows to produce the real-capture evidence table.
