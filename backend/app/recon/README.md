# Optional neural reconstruction

AeroVoxel's primary GPU path uses the real [Stable Fast 3D (SF3D)](https://github.com/Stability-AI/stable-fast-3d) model. It takes one object image, exports a mesh, then voxelizes the mesh's center slice into the existing `(ny=64, nx=128)` mask contract. The mask can be used by the same educational D2Q9 LBM endpoint as an uploaded silhouette.

## Setup on an RTX 3050

Use a CUDA-enabled PyTorch build appropriate for the installed NVIDIA driver, then install the optional dependencies and official SF3D checkout:

```powershell
cd backend
python -m venv recon-venv
recon-venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-recon-cuda126.txt
git clone https://github.com/Stability-AI/stable-fast-3d ..\stable-fast-3d
pip install -r ..\stable-fast-3d\requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The model weights are downloaded by Hugging Face on first run and are deliberately not committed. SF3D's official documentation reports about 6 GB VRAM for its default run; this laptop exposes 4 GB, so the 512 texture setting and batch size 1 are a constrained experiment, not a guarantee of fit. The CUDA requirements file deliberately installs a CUDA-enabled wheel instead of letting generic PyPI resolution select CPU-only PyTorch. The runner uses CUDA autocast FP16 and records peak allocated VRAM when CUDA is active. Set `AEROVOXEL_SF3D_REPO` if the official checkout is stored elsewhere.

## Run and integrate

```powershell
python scripts/run_recon_demo.py C:\path\to\phone-photo.jpg --output-dir data/recon_outputs\R1 --register-for-api
```

The output contains `mesh.glb`, `mask.npy`, and `result.json`. With `--register-for-api`, the validated mask is copied into the existing upload-job storage and the result includes a UUID for `/api/simulate/simple`. This is the L1 integration boundary; the browser does not launch a multi-minute GPU job.

The engine-selection boundary currently has one real 3D implementation, SF3D.
The existing OpenCV upload pipeline is retained as an explicit 2D silhouette
fallback, never relabeled as 3D. TripoSR was evaluated as an alternative but
its official default also reports about 6 GB VRAM, so it is not a credible
fallback for this 4 GB device; adding a second unverified heavy backend would
increase setup risk without removing the host driver blocker.

If CUDA is unavailable, `--device cpu` is supported for smoke testing only and is not a claim that CPU reconstruction is practical. SF3D's CPU backend is a real computation but is not treated as a production fallback on a 16 GB Windows laptop. If SF3D or trimesh is missing, the command fails with an actionable message rather than falling back to a fake mesh.

## Diagnostics and reproducible evaluation

```powershell
python scripts/check_recon_environment.py
python scripts/evaluate_recon.py app/recon/recon_manifest.example.csv
```

The evaluator requires a typed CSV (`PHONE_CAPTURE`, `PUBLIC_DATASET`, or
`SYNTHETIC_EVALUATION`) and writes per-case JSON/CSV records. Missing inputs,
missing dependencies, and model failures remain failures; they are never
converted into benchmark successes. Use `backend/data/recon_inputs/` for
consented captures and keep that directory gitignored.

## Limits

Neural reconstruction is best-effort prototype geometry, not metrology-grade 3D scanning. Single-view depth is ambiguous; transparent, reflective, occluded, blurred, and cluttered objects can fail. Flow remains educational LBM, not certified CFD. Do not commit private phone photos or downloaded model weights.
