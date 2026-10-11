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

Latest host check (2026-09-05): Windows PnP reports the RTX 3050 as
`OK` with `CM_PROB_NONE`, and `nvidia-smi` reports driver 616.56 and 4096 MiB
of VRAM. The active project interpreter is still `torch 2.11.0+cpu`; installing
the isolated CUDA wheel was attempted through pip, curl, Windows BITS, and a
Python 3.11 `uv` environment with extended retries. The 2.4–2.6 GB transfer
repeatedly failed with TLS/security or stream/extraction errors. Do not describe
CUDA or SF3D as validated until the CUDA wheel is installed and a real CUDA
tensor operation succeeds.

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

The evaluator requires a typed CSV (`REAL_PHONE_CAPTURE`, `PHONE_CAPTURE`, `PUBLIC_DATASET`, or
`SYNTHETIC_EVALUATION`) and writes per-case JSON/CSV records. Missing inputs,
missing dependencies, and model failures remain failures; they are never
converted into benchmark successes. Use `backend/data/recon_inputs/` for
consented captures and keep that directory gitignored. The committed synthetic
manifest is explicitly `SYNTHETIC_EVALUATION`; it must not be reported as five
real phone captures.

### Supplying one genuine phone photograph

To run one physical-phone image, transfer the original JPG/PNG/WEBP file to
the ignored `backend/data/recon_inputs/` directory. If your local folder is
named `recon_input` (singular), use that exact name in the command. Use a clear, well-lit image
with one object fully visible against a simple background; do not include
private personal data in the frame. From `backend`, run:

```powershell
python scripts/prepare_phone_manifest.py `
  --image data\recon_inputs\phone_capture_01.jpg `
  --output ..\evaluation_outputs\phone_capture_01\manifest.csv
python scripts/evaluate_phone_capture.py `
  ..\evaluation_outputs\phone_capture_01\manifest.csv `
  --engine auto --device cpu `
  --output-dir ..\evaluation_outputs\phone_capture_01
```

The helper verifies that the file is readable, records SHA-256 and dimensions,
and labels it `REAL_PHONE_CAPTURE`; it never converts the image into synthetic
or public-dataset data. The evaluator runs reconstruction and LBM for every
listed case and writes `results.json`, `results.csv`, and validated artifacts.
Each case must have a distinct SHA-256; duplicate copies are rejected. To
prepare four additional photographs later, pass five distinct image paths and
five case IDs to the same helper; each case is evaluated independently.

```powershell
python scripts/run_recon_demo.py `
  data\recon_inputs\phone_capture_01.jpg `
  --engine auto --device cpu `
  --output-dir ..\evaluation_outputs\phone_capture_01_end_to_end `
  --register-for-api
```

Then POST the printed `job_id` to `/api/simulate/simple` with
`{"wind_speed":15,"wind_angle_deg":0}`. This is the only remaining user action
needed to produce a genuine phone-capture result; provenance cannot be created
by software without the physical image.

The verified local CPU run on 2026-09-04 produced 5/5 successful cases. Every
case produced a watertight GLB with finite, non-degenerate geometry and a
non-empty `(64,128)` solver mask. The five-case wall time was 52.026 seconds
in the first cached run; later runs are machine-load dependent. These are
pipeline/output-validity results, not a ground-truth reconstruction-quality
benchmark.

### Public real-world evaluation data

The repository does not contain five consented phone captures. Pix3D is the
separate public real-world evaluation set; cases are labeled `PUBLIC_DATASET`,
never `PHONE_CAPTURE`. The official Pix3D release is a 3.6 GB archive and is
licensed CC BY 4.0. Acquire it from the
[official Pix3D project](https://pix3d.csail.mit.edu/) and follow the dataset
attribution terms.

The manifest builder selects readable, non-truncated and non-occluded images,
preferring category diversity. It records stable source record indices,
category, image source, dimensions, image SHA-256, and local paths to available
Pix3D masks, models and voxel annotations. The dedicated evaluator verifies
the checksum, runs reconstruction, validates the serialized mesh, generates
the solver mask, calls the existing simulation API/LBM endpoint, and copies
the resulting mask, velocity and pressure arrays into each case output:

```powershell
cd backend
python scripts/prepare_pix3d_manifest.py --pix3d-root data\pix3d_public_subset --output ..\evaluation_outputs\pix3d_public_subset\manifest.csv --n 5
python scripts/evaluate_public_dataset.py ..\evaluation_outputs\pix3d_public_subset\manifest.csv --engine auto --device cpu --output-dir ..\evaluation_outputs\pix3d_public_evaluation
```

The result bundle includes `results.json`, `results.csv`, validated GLBs,
depth/mask artifacts, and per-case LBM arrays and metrics. Failures are isolated
per sample. No 3D ground-truth score is computed: the current Depth Anything
V2 output is a view-conditioned depth relief, not a canonical reconstruction
aligned to Pix3D voxel/model coordinates. No 3D IoU/CD/EMD score is claimed
without valid coordinate alignment.

Verified local run: five cases across bed, bookcase, chair, desk, and misc;
all five produced valid GLBs, solver masks, and completed LBM simulations on
CPU with Depth Anything V2 Small. These are pipeline/output-validity results,
not reconstruction-quality benchmark scores. Machine-specific timings and
metrics are in the ignored local
`evaluation_outputs/pix3d_public_evaluation/results.json`. The downloaded
subset and generated outputs are not committed.

### Alternative-model decision record

TripoSR was reviewed as the closest single-image alternative to SF3D. Its
official repository describes a real image-to-3D mesh pipeline and an MIT
license, but also reports roughly 6 GB default VRAM; that does not remove the
current 4 GB constraint, and no unsupported memory claim is made here. Hunyuan3D
was not selected because its official 2.1 requirements are substantially above
this laptop's VRAM and its community license is more restrictive. Point-E and
Shap-E were not selected as the primary path because their older output/runtime
tradeoffs would require a separate point-cloud or implicit-surface conversion
pipeline. Depth Anything V2 Small remains the executable real-ML alternative:
it is explicitly a monocular depth-conditioned relief reconstruction, not a
claim of full hidden-surface recovery or an SF3D-equivalent model.

Relevant upstream references: [TripoSR](https://github.com/VAST-AI-Research/TripoSR),
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1),
[Pix3D](https://github.com/xingyuansun/pix3d).

## Limits

Neural reconstruction is best-effort prototype geometry, not metrology-grade 3D scanning. Single-view depth is ambiguous; transparent, reflective, occluded, blurred, and cluttered objects can fail. Flow remains educational LBM, not certified CFD. Do not commit private phone photos or downloaded model weights.
