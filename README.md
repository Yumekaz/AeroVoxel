# AeroVoxel

**Educational virtual wind tunnel from smartphone imagery** — lightweight Lattice Boltzmann simulation, interactive browser visualization, and a trained ML surrogate for fast educational drag proxies.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=black)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Portfolio-lightgrey)](#license)

---

## Why this exists

Physical wind tunnels and production CFD are expensive, slow to set up, and hard to approach for early learning. AeroVoxel is a **browser-first prototype** that turns a simple photo (or a built-in demo shape) into an interactive airflow view: pressure coloring, streamlines, wake cues, and educational force-style metrics — running on a normal laptop CPU.

It is built as a **full system**, not a single notebook: computer vision → simulation domain → Lattice Boltzmann (LBM) → Three.js viewer, plus an optional **machine-learning surrogate** trained on this project’s own LBM labels.

> **Scope:** Educational and prototype-grade. **Not** certified CFD, **not** a substitute for professional wind-tunnel testing, and **not** perfect AI-based 3D reconstruction from video.

## AI / ML modules

1. **Educational LBM surrogate (CPU)** — a trained, evaluated scikit-learn model maps masks to this project's `cd_force_proxy` labels; metrics and baselines are in [`docs/ai_metrics.md`](docs/ai_metrics.md).
2. **Neural 3D reconstruction (hardware-aware)** — SF3D is the preferred GPU adapter when its runtime and VRAM requirements are met; Depth Anything V2 Small is a real CPU/CUDA alternative that produces a depth-conditioned mesh and center-slice `mask.npy`; OpenCV remains the explicit final 2D fallback. See [`backend/app/recon/README.md`](backend/app/recon/README.md).

Neural reconstruction is best-effort prototype geometry, not metrology-grade scanning. The current interactive path remains 2D CPU LBM; no certified CFD or perfect video-to-3D claim is made.

---

## Highlights

| Capability | What you get |
|------------|----------------|
| **Interactive wind tunnel UI** | Three.js viewer with streamlines, pressure fields, and clear data-source labels |
| **Demo library** | Car, drone, airfoil, circular cylinder — precomputed educational 2D flow fields |
| **Image / video upload** | OpenCV silhouette extraction and optional scale-card detection |
| **Live 2D physics** | D2Q9 Lattice Boltzmann on CPU from an uploaded mask |
| **Offline 3D sample** | Optional coarse D3Q19 sphere dataset (generated offline; shown only when assets exist) |
| **ML surrogate** | Trained models map a 2D mask → educational `cd_force_proxy` in milliseconds, with baselines and held-out metrics |
| **Resilient demo path** | Cached cases and offline fallback so presentations do not depend on a perfect network |

---

## System overview

```text
  Phone image / demo shape
            │
            ▼
   OpenCV mask + scale cues
            │
            ├──────────────────► Live 2D LBM (CPU) ──┐
            │                                         │
            ├──────────────────► Cached flow fields ──┼──► Browser wind tunnel
            │                                         │         (Three.js)
            └──────────────────► ML surrogate ────────┘
                                 (fast Cd proxy only)
```

**Shared contract:** velocity, pressure, and mask arrays use one schema so the UI does not care whether data came from cache, a live solve, or offline 3D precompute.

---

## Tech stack

| Layer | Choices |
|-------|---------|
| Frontend | React, Vite, Three.js, TypeScript |
| Backend | FastAPI, NumPy, OpenCV |
| Simulation | Custom D2Q9 (2D) and D3Q19 (3D) Lattice Boltzmann solvers |
| Machine learning | scikit-learn surrogates (ablation: linear, MLP, tree ensembles, …) trained on in-house LBM labels |
| Packaging | Local venv + npm; no cloud dependency for the core demo |

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env           # optional — default API http://127.0.0.1:8000
npm run dev
```

Open **http://localhost:5173** → try a demo object or upload an image.

### 3. Optional assets

```bash
cd backend
venv\Scripts\activate
python -m app.services.cache_generator        # 2D demo flow fields
python -m app.services.cache_generator_3d      # sphere 3D cache (~1–3 min CPU)
```

The Sphere 3D entry appears in the app only when the corresponding `.npy` files exist, so the demo list never offers a broken case.

---

## Machine learning surrogate

AeroVoxel includes a **real training pipeline** (not an external AI API wrapper):

1. Generate synthetic obstacle masks  
2. Label them with this project’s 2D LBM (`cd_force_proxy`)  
3. Train and ablate models (mean, linear geometry, MLP, gradient boosting, …)  
4. Serve the **best held-out** model via API and the UI button **Predict Cd (ML surrogate)**

| | |
|--|--|
| **Predicts** | Educational force proxy aligned with *this* LBM setup |
| **Does not predict** | Certified drag coefficients or full 3D industrial CFD fields |
| **Metrics** | See committed snapshot [`backend/app/ml/artifacts/last_eval.json`](backend/app/ml/artifacts/last_eval.json) |
| **Typical scale** | ~800 training samples on a 64×32 grid (laptop-friendly) |

```bash
cd backend
python scripts/generate_ml_dataset.py --n 800 --seed 42
python scripts/train_surrogate.py --seed 42
python scripts/eval_surrogate.py
```

Without a trained checkpoint, the API returns **503** with train instructions; the UI disables the predict button.

---

## Evaluation & reproducibility

Scripts under `backend/scripts/` support educational validation and robustness checks (reference shapes, grid sensitivity, weak inputs, phone-like synthetic photos). Outputs write to gitignored `evaluation_outputs/`.

```bash
cd backend
python scripts/run_evaluation.py --live-2d --try-sphere-cache
python scripts/run_grid_study.py
python scripts/run_failure_tests.py
python scripts/run_phone_photo_tests.py
```

For a real-world public-data substitute when consented phone captures are not
available, use the optional Pix3D preparation/evaluation path. Pix3D is a
public real image/shape dataset and must remain labeled `PUBLIC_DATASET`; it
must not be described as phone capture data:

```bash
cd backend
python scripts/prepare_pix3d_manifest.py \
  --pix3d-root /datasets/pix3d \
  --output app/recon/recon_manifest.pix3d.csv \
  --n 5
python scripts/evaluate_recon.py app/recon/recon_manifest.pix3d.csv \
  --engine depth_anything --device cpu \
  --output-dir ../evaluation_outputs/pix3d_depth_anything
```

The project intentionally does not bundle the several-gigabyte dataset or
private captures. The preparation script verifies local files and records only
cases that actually exist. See [`backend/app/recon/README.md`](backend/app/recon/README.md)
for the alternative-model decision record and current GPU limitations.

These runs document **operating limits** (coarse grids, educational metrics, low-Re offline 3D) rather than claiming mesh-independent engineering accuracy.

---

## API (summary)

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Health check |
| `GET` | `/api/demo-cases` | Available demos (asset-gated) |
| `GET` | `/api/flow-field/{case_id}` | Flow metadata + array URLs |
| `POST` | `/api/upload` | Image/video → mask pipeline |
| `POST` | `/api/simulate/simple` | Live 2D LBM |
| `GET` | `/api/surrogate/status` | Checkpoint availability |
| `POST` | `/api/surrogate/predict` | Fast educational Cd-proxy |
| `GET` | `/api/recon/capability` | Report optional SF3D/CUDA capability without loading weights |
| `GET` | `/api/recon/diagnostics` | Non-destructive runtime, NVIDIA, and display-device diagnostics |

---

## Project layout

```text
AeroVoxel/
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/          # REST API
│   │   ├── solvers/         # LBM 2D / 3D
│   │   ├── ml/              # Surrogate train / infer
│   │   ├── services/        # Cache generators
│   │   └── assets/flow/     # Demo flow fields
│   └── scripts/             # Eval, ML, robustness
└── frontend/
    └── src/                 # React + Three.js app
```

---

## Design principles

1. **Honest labels** — cached vs live vs offline 3D vs ML proxy are visible to the user.  
2. **One data contract** — UI stays stable as solvers improve.  
3. **Laptop-first** — live work stays 2D CPU; heavy 3D is optional offline precompute.  
4. **ML with ground truth** — the surrogate is judged against this project’s LBM, with baselines.  
5. **Demo reliability** — missing optional assets are hidden, not broken links.

---

## Limitations (read this)

- Metrics are **educational** (heuristics and force proxies), not certified \(C_d\) / \(C_l\).  
- Geometry from upload is a **silhouette / template path**, not full multi-view 3D reconstruction.  
- Grids are **coarse** by design (interactive teaching, not design sign-off).  
- Offline sphere cases run at **low Reynolds number**; do not compare casually to high-Re literature drag charts.  
- The ML model does **not** generalize to arbitrary real-world aerodynamics outside this labeling pipeline.

---

## License

Portfolio and educational prototype. Use and adapt with attribution; do not present results as certified engineering analysis.
