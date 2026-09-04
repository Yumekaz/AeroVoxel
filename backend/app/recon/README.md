# Optional neural reconstruction

AeroVoxel's primary GPU path uses the real [Stable Fast 3D (SF3D)](https://github.com/Stability-AI/stable-fast-3d) model. It takes one object image, exports a mesh, then voxelizes the mesh's center slice into the existing `(ny=64, nx=128)` mask contract. The mask can be used by the same educational D2Q9 LBM endpoint as an uploaded silhouette.

## Setup on an RTX 3050

Use a CUDA-enabled PyTorch build appropriate for the installed NVIDIA driver, then install the optional dependencies and official SF3D checkout:

```powershell
cd backend
python -m venv recon-venv
recon-venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-recon.txt
git clone https://github.com/Stability-AI/stable-fast-3d ..\stable-fast-3d
pip install -r ..\stable-fast-3d\requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The model weights are downloaded by Hugging Face on first run and are deliberately not committed. For a 4 GB card, use batch size 1 and the default 512 texture resolution. The runner uses CUDA autocast FP16 and records peak allocated VRAM when CUDA is active.

## Run and integrate

```powershell
python scripts/run_recon_demo.py C:\path\to\phone-photo.jpg --output-dir data/recon_outputs\R1 --register-for-api
```

The output contains `mesh.glb`, `mask.npy`, and `result.json`. With `--register-for-api`, the validated mask is copied into the existing upload-job storage and the result includes a UUID for `/api/simulate/simple`. This is the L1 integration boundary; the browser does not launch a multi-minute GPU job.

If CUDA is unavailable, `--device cpu` is supported for smoke testing only and is not a claim that CPU reconstruction is practical. If SF3D or trimesh is missing, the command fails with an actionable message rather than falling back to a fake mesh.

## Limits

Neural reconstruction is best-effort prototype geometry, not metrology-grade 3D scanning. Single-view depth is ambiguous; transparent, reflective, occluded, blurred, and cluttered objects can fail. Flow remains educational LBM, not certified CFD. Do not commit private phone photos or downloaded model weights.
