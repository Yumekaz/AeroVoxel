# AeroVoxel local verification - 2026-10-13

## Implementation

Selected visual direction: Open Tunnel (second design). Existing dark/cyan
identity retained, with horizontal demo picker, wide viewer, camera controls,
compact dock, collapsible results, mobile reflow and reduced-motion handling.
Flow paths and geometry use the same coordinate mapping; decorative offline
previews are labeled separately. Simulation parameter changes require rerunning
custom LBM; ML remains a fixed-condition proxy.

Reconstruction now preserves XY proportions, leaves airflow margins, validates
SF3D serialized geometry, respects requested device/VRAM and preserves failed
auto attempts. A bounded mask import endpoint connects offline outputs to the
browser without installing a GPU inference service. Characteristic length is
shared between label generation and the live API; different grid/run/inlet
conditions remain explicit. Existing surrogate weights and dataset are retained.

## Executed evidence

- Frontend production build and ESLint: passed. Vite reports the existing large
  JavaScript-bundle advisory; no build errors.
- Backend regression suite: 36 passed in 15.46 seconds in the final run (including the original photo-to-LBM path and display/field agreement checks).
  CPU-only environment assumptions were replaced with mocked capability checks.
- Original weak-input robustness matrix: F1-F5 all passed, preserved in a fresh
  evaluation directory rather than overwriting previous evidence.
- Real Depth Anything V2 Small CPU inference on SYNTH_01-SYNTH_05: 5/5 succeeded.
  Every case produced serialized mesh and normalized mask artifacts, completed
  LBM, and saved finite mask/velocity/pressure arrays.
- Five-case reconstruction wall time including first model download/load:
  102.367 seconds. This is machine/load-specific and not an SLO.
- Browser checks: demo selection, side/perspective, reset, pause/play, results
  collapse/expand, original surrogate inference, reconstruction mask import,
  geometry preview and real CPU simulation completion.
- Mobile viewport 390x844: document width 390, controls/results reflow vertically.

Raw generated evidence remains ignored under
`evaluation_outputs/cpu_e2e_20261013_attempt1/` and
`evaluation_outputs/robustness_20261013/`. Source code and small documentation
are separate from model caches and personal input data.

## Detailed model and airflow refinement

The car now uses the locally served Khronos CarConcept model by Eric Chadwick /
Darmstadt Graphics Group GmbH under CC BY 4.0. Source, modifications and license
are in `frontend/public/models/ATTRIBUTION.md`. Original metadata remains in the
asset. Branding/color and emissive maps are removed; normal, occlusion and roughness
maps preserve their UV transforms. `frontend/scripts/prepare-car-asset.py` reproduces
the derivative from the official source GLB. The asset is 11,473,640 bytes and loads
on first car selection, with a visible loading state and a basic fallback on failure.

`frontend/scripts/export-demo-masks.mjs` projects the same car/drone geometry into
the 128x64 lattice with conservative 4x sampling. The car has 353 solid cells and
the drone 182. Both cached CPU fields were regenerated for 800 D2Q9 steps at
tau 0.6 and inlet velocity 0.08. `shared/demoGeometry.json` records placement,
occupancy and asset hash; `visual_demo_generation.json` records array hashes,
conditions and educational proxy estimates. Geometry checks reconstruct and compare
these projections without modifying published data. All six running API array
responses matched their current on-disk SHA-256 hashes.

Trials 1-5 are preserved under `evaluation_outputs/visual_fields_20261013_attempt*/`;
attempt 5 is the published final asset/field record. Visual checking exposed an old
browser-cached field even though the API served current files. Array requests now
revalidate (`cache: no-cache`), and capture 37 verifies the refreshed car mask.
Earlier capture 36 records the rejected stale field and remains preserved.

Airflow uses soft camera-facing ribbons, fluid-only velocity interpolation and
midpoint integration; seeding density near the body improves visibility without
changing computed velocities. Geometry lighting, reflections and floor treatment
are refined. Static shadows update when geometry/overlays change instead of every
animation frame. These changes do not turn the field into a full 3D car simulation.

Final visual evidence includes matched car flow (38), mobile reflow (43), and the
selected-concept comparison (42), in the chat's `frontend-audit` output directory.
Mobile document width was 390 at a 390x844 viewport; car, legend and controls fit.
Production build, lint, geometry comparison and backend suite passed. Vite still
reports the large Three.js application chunk; FastAPI's test client reports an
upstream deprecation warning. Neither produced a failed check.
Final wing-section shading and drone materials are shown in captures 48/49;
capture 50 leaves the final car selected with the results panel visible.
Pause/play state and the code's frozen integration/history were checked. Two pixel
comparison attempts were inconclusive due to camera settling after view/viewport
changes; raw captures 39/40 and 44/45 are preserved without a pixel-exact claim.

## Limits

Five physical-phone photographs have not been supplied. Synthetic inputs are
never labeled real phone data. CUDA/SF3D inference, physical-device UX,
screen-reader verification and ground-truth reconstruction quality were not
performed. SF3D adapter tests use a model stub plus real geometry export, not
GPU weights. CPU neural evidence is real pretrained Depth Anything inference.
Automated browser captures sometimes lagged state changes; rejected captures are
retained. The browser's FPS display sampled 1-6 FPS in this automation session,
including lightweight scenes, so sustained animation performance is not certified
by these captures. Physical-device performance and slow-network loading remain
unmeasured.

## Real web photographs follow-up

The subsequent five-case public-photo CPU run completed all reconstructions and
LBM solves. It exposed and corrected independent X/Y stretching in the depth mesh
builder. The full regression suite is now 42 passing tests. Foreground/mesh ratio
agreement is within about 1%; 3/5 meshes are watertight, and thin/reflective object
parts and shadows remain quality limitations. Physical-phone capture/performance
and CUDA/SF3D validation remain outstanding. Source attribution, raw evidence
locations, actual performance observations and reproduction commands are in
[the web-photo CPU report](web_photo_cpu_2026-10-13.md).
