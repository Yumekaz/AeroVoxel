# Optional neural reconstruction

AeroVoxel uses a hardware-aware reconstruction engine chain:

1. SF3D when its full runtime and hardware are available.
2. Depth Anything V2 Small on CPU/CUDA as the practical real-ML alternative. It predicts monocular depth, creates a depth-conditioned thickened mesh, then exports the center-slice mask.
3. OpenCV 2D silhouette only when both ML engines are unavailable.

The first two are real model-based inference paths; the last is explicitly not 3D or ML.

On a CPU-only machine, install the real alternative explicitly:

```powershell
cd backend
pip install -r requirements-recon-cpu.txt
python scripts/run_recon_demo.py C:\path\to\photo.jpg --engine depth_anything --device cpu --output-dir data/recon_outputs\R1
```

The first run downloads `depth-anything/Depth-Anything-V2-Small-hf` from the
Hugging Face Hub. Subsequent requests reuse the in-process model cache. The
alternative is monocular relative-depth reconstruction: it produces a closed,
depth-conditioned relief mesh and a normalized center slice, but it cannot
recover hidden surfaces or metric scale from one image.

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

Latest host check (2026-09-04): Windows PnP reports the RTX 3050 as
`OK` with `CM_PROB_NONE`, and `nvidia-smi` reports driver 616.56 and 4096 MiB
of VRAM. The active project interpreter is still `torch 2.11.0+cpu`; installing
the isolated CUDA wheel was attempted through pip, curl, and Windows BITS but
the 2.6 GB download repeatedly failed with TLS/security transport errors. Do
not describe CUDA or SF3D as validated until the CUDA wheel is installed and a
real CUDA tensor operation succeeds.

## Run and integrate

```powershell
python scripts/run_recon_demo.py C:\path\to\phone-photo.jpg --engine auto --output-dir data/recon_outputs\R1 --register-for-api
```

The output contains `mesh.glb`, `depth.npy`, `depth_preview.png`, `mask.npy`, and `result.json`. `result.json` reports the selected engine, device, fallback reason, timings, mesh validity, serialized-file validity, and mask statistics. With `--register-for-api`, the validated mask is copied into the existing upload-job storage and the result includes a UUID for `/api/simulate/simple`. This is the L1 integration boundary; the browser does not launch a multi-minute GPU job.

The engine-selection boundary keeps SF3D as the preferred real 3D engine,
selects Depth Anything V2 Small when SF3D is unavailable, and only then uses
the existing OpenCV upload path as a 2D fallback. TripoSR was evaluated as an
alternative but its official default also reports about 6 GB VRAM, so it is
not a credible fallback for this 4 GB device; adding a second unverified heavy
backend would increase setup risk without removing the host driver blocker.

If CUDA is unavailable, `--device cpu` runs the real Depth Anything alternative. SF3D's CPU backend is a real computation but is not treated as the practical fallback on a 16 GB Windows laptop. If SF3D or the alternative runtime is missing, selection reaches the explicit OpenCV 2D fallback; that path never claims to be 3D or ML.

## Diagnostics and reproducible evaluation

```powershell
python scripts/check_recon_environment.py
python scripts/evaluate_recon.py app/recon/recon_manifest.example.csv
# After generating the labeled synthetic fixtures:
python scripts/run_phone_photo_tests.py --n 5 --seed 7
python scripts/evaluate_recon.py app/recon/recon_manifest.synthetic.csv --engine depth_anything --device cpu --output-dir ..\evaluation_outputs\depth_anything_five_case_verified
```

The evaluator requires a typed CSV (`PHONE_CAPTURE`, `PUBLIC_DATASET`, or
`SYNTHETIC_EVALUATION`) and writes per-case JSON/CSV records. Missing inputs,
missing dependencies, and model failures remain failures; they are never
converted into benchmark successes. Use `backend/data/recon_inputs/` for
consented captures and keep that directory gitignored. The committed synthetic
manifest is explicitly `SYNTHETIC_EVALUATION`; it must not be reported as five
real phone captures.

The verified local CPU run on 2026-09-04 produced 5/5 successful cases. Every
case produced a watertight GLB with finite, non-degenerate geometry and a
non-empty `(64,128)` solver mask. The five-case wall time was 52.026 seconds
in the first cached run; later runs are machine-load dependent. These are
pipeline/output-validity results, not a ground-truth reconstruction-quality
benchmark.

## Limits

Neural reconstruction is best-effort prototype geometry, not metrology-grade 3D scanning. Single-view depth is ambiguous; transparent, reflective, occluded, blurred, and cluttered objects can fail. Flow remains educational LBM, not certified CFD. Do not commit private phone photos or downloaded model weights.
