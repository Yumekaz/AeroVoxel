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

The current machine now reports the RTX 3050 as healthy through Windows PnP
(`CM_PROB_NONE`), and `nvidia-smi` communicates with it, reporting driver
616.56 and 4096 MiB VRAM. The active project PyTorch remains CPU-only because
the isolated CUDA 12.6 wheel download failed repeatedly with Windows TLS/BITS
security errors. The real CPU alternative is verified; CUDA/SF3D evidence
remains unverified, and the device's 4 GB VRAM is below SF3D's documented
roughly 6 GB default. Install the CUDA requirements after the wheel transfer
is available, then require a successful real CUDA tensor test before running
SF3D.
