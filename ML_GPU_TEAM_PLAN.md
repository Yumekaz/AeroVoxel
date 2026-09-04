# AeroVoxel — GPU / AI–ML Team Plan (RTX 3050)

**Audience:** Teammate with **NVIDIA RTX 3050**  
**Owner of this plan:** share + execute together  
**Repo:** AeroVoxel (browser virtual wind tunnel + LBM + existing educational surrogate)  
**Goal:** Make the project **provably multi-field** (CV + physics sim + systems + **real AI/ML**) so teachers and hiring reviewers classify it as a **proper AI-related project**, not “only a Three.js demo.”

---

## 1. Positioning (how we talk about the project)

### One-line pitch (use this)

> AeroVoxel is an **applied AI + simulation system**: we turn object imagery into a simulation domain, run **Lattice Boltzmann** airflow, visualize results in the browser, and ship **trained ML models** (surrogate ± optional neural 3D / segmentation) with **metrics and baselines**.

### What we already have (do not re-do)

| Component | Status | Role in “AI story” |
|-----------|--------|---------------------|
| OpenCV upload / silhouette / scale cues | Done (CPU) | Classical CV pipeline + data entry |
| Live 2D LBM + cached demos + optional offline 3D LBM | Done (CPU) | Physics **label factory** + ground truth for ML |
| sklearn surrogate (mask → `cd_force_proxy`), N≈800, ablations, API + UI button | Done | **Core ML proof already** |
| Eval scripts (grid study, failure tests, phone-like tests) | Done | Engineering rigor |
| Public README / About (honest scope) | Done | No fake “certified CFD” / no fake “recon solved” |

### What the 3050 unlocks (your job focus)

| Workstream | Why it proves AI | Needs 3050? |
|------------|------------------|-------------|
| **A. Neural 3D reconstruction path** (image/video → mesh → voxel/mask → existing LBM/viewer) | Clear **deep learning / vision** module | **Yes** (local CUDA) |
| **B. Stronger ML package** (optional CNN / better reports / more data) | Stronger ML metrics narrative | Helpful, not mandatory |
| **C. Learned segmentation** (optional U-Net vs OpenCV) | Classic CV-ML ablation | Helpful |
| **D. Integration + eval + write-up** | Proof for teachers/hiring | CPU OK after you export artifacts |

**Rule:** AI modules must **plug into AeroVoxel**, not live as a disconnected Colab demo.

---

## 2. Success criteria (what “proved” means)

Teachers / hiring people should be able to open the repo and see:

1. **Trained models** (weights or clear train scripts + committed metrics JSON)  
2. **Dataset story** (how labels/geometry were produced)  
3. **Baselines + numbers** (tables, not vibes)  
4. **End-to-end path** in the product (or documented offline → import path)  
5. **Honest limits** (prototype / educational physics; recon is best-effort)

### Minimum bar for “this is an AI project”

- [x] Surrogate already satisfies **minimum ML** if presented first  
- [ ] **Plus GPU track:** at least **one** neural recon demo integrated or clearly imported into the tunnel  
- [ ] One **metrics doc** that lists all AI modules side by side  

### Strong bar (aim here)

- [ ] Recon on **≥5** real phone captures (good lighting, orbit if multi-view)  
- [ ] Geometry quality notes (success/fail cases)  
- [ ] Mesh → simulation domain → **same** flow viewer  
- [ ] Optional: segmentation IoU table (OpenCV vs learned)  
- [ ] Short **AI chapter** screenshots + table for the report  

---

## 3. Hardware notes (RTX 3050)

| Spec reality | Implication |
|--------------|-------------|
| CUDA NVIDIA GPU | Use **PyTorch CUDA** builds, not CPU-only for recon |
| **4 GB vs 6 GB VRAM** (check your SKU) | Prefer lightweight recon models; lower resolution; batch size 1 |
| 3050 laptop may thermal throttle | Night runs OK; don’t promise real-time recon in UI |
| Windows | CUDA toolkit + recent NVIDIA driver; use venv/conda |

**Check once:**

```text
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `cuda.is_available()` is false, fix drivers/PyTorch **before** any recon work.

**VRAM hygiene:** close games/browsers with many tabs; use `torch.cuda.empty_cache()` between runs; prefer 256–512 px input first.

---

## 4. Workstream A — Neural 3D reconstruction (primary GPU task)

### Objective

Add a **documented AI path**:

```text
Phone image (or few frames)
    → neural reconstruction model (GPU)
    → mesh / point cloud
    → voxel or silhouette domain for AeroVoxel
    → existing LBM (2D slice or offline) + Three.js viewer
```

This proves **AI beyond sklearn tabular surrogate**.

### Recommended approach (practical on 3050)

| Priority | Option | Notes |
|----------|--------|--------|
| **1 (recommended)** | Single-image feed-forward mesh (e.g. InstantMesh / TripoSR / similar **current** open weights) | Fastest path to a mesh on consumer GPU |
| **2** | Multi-view classical + light neural cleanup | More “video,” more setup (COLMAP); use if single-image quality is poor |
| **3** | Full NeRF / heavy 3DGS training per object | **Avoid** as main path — slow, overkill for integration |

**Pick one primary stack and stick to it.** Do not try three recon frameworks.

### Deliverables (files / artifacts)

Propose this layout (adjust names if needed):

```text
backend/
  app/
    recon/                  # OR tools/recon/ if cleaner
      README.md             # how to run on 3050
      export_to_aerovoxel.py
  data/
    recon_inputs/           # gitignored phone photos (user’s)
    recon_outputs/          # gitignored meshes / previews
backend/scripts/
  run_recon_demo.py         # CLI: image → mesh → mask/voxel export
docs/ or backend/app/ml/artifacts/
  recon_eval_notes.md       # success/fail table (can be committed if no private data)
```

**Git rules (important):**

- Do **not** commit large weights if multi‑100MB unless team agrees; prefer script + download instructions  
- Do **not** commit personal phone photos without consent; use gitignored folders  
- **Do** commit: code, small example mesh if tiny, metrics/notes, README section  

### Integration contract with AeroVoxel

Minimum viable integration (choose one):

| Level | What | Good enough for demo? |
|-------|------|------------------------|
| **L1** | Export **center-slice binary mask** from mesh → existing `POST /api/simulate/simple` | Yes |
| **L2** | Export **voxel grid** / occupancy for offline 3D path | Better story |
| **L3** | UI button “Run AI recon” calling local GPU service | Nice; only after L1 works |

**Do L1 first.** UI is optional polish.

### Evaluation (mandatory — this is the “proof”)

Create a small table (markdown) for **≥5** cases:

| Case ID | Input type | Recon success? | Notes (holes, scale, junk geometry) | Downstream sim ran? |
|---------|------------|----------------|-------------------------------------|---------------------|
| R1 | … | Y/N | … | Y/N |

Also record:

- Wall time per recon on 3050  
- Peak VRAM if easy (`nvidia-smi`)  
- Failure modes (blur, single view ambiguity, transparent objects)

**Honesty line (always):**  
Neural recon is **best-effort prototype geometry**, not metrology-grade 3D scanning, and flow remains **educational LBM**, not certified CFD.

---

## 5. Workstream B — Strengthen existing surrogate ML (optional, high ROI if time)

Already strong for “we do ML.” GPU can help only if you move to a **small CNN** on masks.

| Task | Owner | Priority |
|------|-------|----------|
| Keep sklearn pipeline as baseline | Either | Keep |
| Optional: PyTorch CNN on 64×32 masks, train on 3050 | GPU teammate | P2 |
| Prove CNN or best model **beats** `linear_geom` / current primary on held-out set | GPU teammate | P2 |
| Update `last_eval.json` + short note | Either | P2 |

**Do not delete** the current surrogate; add comparison rows.

---

## 6. Workstream C — Learned segmentation (optional CV–ML)

| Task | Why |
|------|-----|
| Label 50–100 masks (synthetic + few real) | Dataset |
| Train small U-Net / segmentation model on 3050 | Clear “deep learning CV” |
| Table: OpenCV silhouette IoU vs model IoU on held-out images | Ablation teachers love |

Only if Workstream A is done or blocked.

---

## 7. Role split

| Person | Machine | Owns |
|--------|---------|------|
| **GPU teammate (3050)** | NVIDIA laptop | Recon env, model runs, mesh export, recon eval table, optional CNN/seg |
| **Main repo / CPU side** | Ryzen iGPU machine | AeroVoxel app stability, LBM, surrogate scripts, API/UI integration of exports, public README honesty, merge PRs |
| **Both** | — | Demo script for viva: ML first, then tunnel; limits last |

### Communication

- One branch per workstream: `feat/neural-recon`, `feat/seg-ml`  
- PR into `main` only with: run instructions + sample output + no huge binaries  
- Weekly: 5-case recon table updated  

---

## 8. Suggested timeline (2–3 weeks, adjustable)

### Week 1 — Environment + first mesh

- [ ] CUDA + PyTorch verified on 3050  
- [ ] One recon model running on **1 clean photo** of a simple object  
- [ ] Export mesh (obj/glb/ply)  
- [ ] Export **2D mask** compatible with AeroVoxel simulate API  
- [ ] Document commands in `recon/README.md`  

### Week 2 — Integration + multi-case proof

- [ ] Script: image → mask/voxel → save under agreed paths  
- [ ] Run LBM or cached path on recon-derived domain  
- [ ] **5+ case** success/fail table  
- [ ] Optional: thin UI “load recon export”  

### Week 3 — Polish for teachers / hiring

- [ ] README subsection: **AI modules** (surrogate + neural recon)  
- [ ] Metrics snapshot committed  
- [ ] 60-second demo script (below)  
- [ ] Failure collage (2–3 bad cases) — shows maturity  

---

## 9. Demo script (60–90 seconds for teachers)

1. **“AI module 1 — Surrogate”**  
   - Show train/eval scripts + metrics table (MAE vs baselines)  
   - Click **Predict Cd (ML surrogate)** in UI  
2. **“AI module 2 — Neural recon (GPU)”**  
   - Show phone image → mesh preview → mask/voxel  
3. **“Physics + systems”**  
   - Run / show LBM or flow viewer on that geometry  
4. **“Limits (10 seconds)”**  
   - Educational flow metrics; recon best-effort; not production CFD  

**Order matters:** ML/AI first, pretty smoke second.

---

## 10. README / public wording (when you merge)

Add a section like:

```text
## AI / ML modules
1. Educational LBM surrogate (CPU) — trained on in-house simulation labels
2. Neural 3D reconstruction (GPU, optional) — image to mesh to simulation domain
```

**Never claim:**

- Certified aerodynamics  
- Solved arbitrary video → perfect 3D  
- That D3Q19 is the only/main interactive path if live path is still 2D  

---

## 11. Definition of done (checklist for the team)

### Must have

- [ ] 3050 runs at least one neural recon model successfully  
- [ ] Automated or scripted export into AeroVoxel domain (mask minimum)  
- [ ] ≥5 documented cases (pass/fail)  
- [ ] Surrogate metrics still present and presented as AI #1  
- [ ] Honest limits written once in README  

### Nice to have

- [ ] UI entry point for recon exports  
- [ ] CNN/seg ablation  
- [ ] Short screen recording for portfolio  

---

## 12. Risk list (read before promising dates)

| Risk | Mitigation |
|------|------------|
| 4 GB VRAM OOM | Smaller input size; lighter model; CPU offload last resort |
| Recon mesh unusable for LBM | Always have silhouette fallback; document fail |
| Scope explosion (NeRF everything) | Single-image mesh only for v1 |
| Overclaim to teachers | Use demo script + limits slide |
| Huge weights in git | Scripts + download URL, not 2 GB commits |

---

## 13. Quick start for the 3050 teammate (Day 0)

1. Clone AeroVoxel, read root `README.md` (product honesty).  
2. Run **existing** surrogate path on CPU once (understand labels):  
   `backend/scripts/generate_ml_dataset.py`, `train_surrogate.py`, `eval_surrogate.py`  
3. Install NVIDIA driver + PyTorch CUDA; verify `torch.cuda.is_available()`.  
4. Create `feat/neural-recon` branch.  
5. Get **one** mesh from **one** photo; stop and sync with teammate before building UI.  
6. Only then automate export → AeroVoxel mask.  

---

## 14. Bottom line for both of you

| Field | Proof in AeroVoxel |
|-------|-------------------|
| **Systems / full-stack** | FastAPI + React + Three.js tunnel |
| **Computer vision** | OpenCV (+ optional neural recon / seg on 3050) |
| **Scientific computing** | Custom LBM solvers |
| **Machine learning** | Trained surrogate + metrics (**already**) |
| **Deep learning on GPU** | Neural recon on 3050 (**this plan**) |

Together that is a **multi-field, AI-centered applied project** — not a single-course toy — **if** the GPU work is integrated and measured, not a loose Colab screenshot.

---

**Document version:** 1.0  
**For:** RTX 3050 teammate execution  
**Out of scope for this plan:** certified CFD, production metrology scanning, training foundation models from scratch.
