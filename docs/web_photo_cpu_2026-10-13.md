# Real web-photo CPU evaluation — 2026-10-13

Five genuine object photographs from Wikimedia Commons completed the existing
Depth Anything V2 Small CPU → serialized relief mesh → voxel center slice → CPU
LBM pipeline. All five produced finite simulation arrays and usable binary masks
with fluid on every domain boundary. This is an execution result, not a claim of
accurate full-object reconstruction or validated aerodynamics.

Inputs are explicitly `PUBLIC_DATASET`. They are neither synthetic fixtures nor
verified physical-phone captures. The original images, metadata, hashes, timings,
meshes, depth maps and simulation arrays remain in ignored evaluation directories.

## Sources and results

| Photograph | Author / selected license | Reconstruction time | Solid cells | Mesh closed? |
| --- | --- | ---: | ---: | --- |
| [Toy car](https://commons.wikimedia.org/wiki/File:Hot_Wheels_Modellauto.jpg) | Stefan-Xp / CC BY-SA 3.0 | 23.233 s | 1011 | Yes |
| [Rubik cube](https://commons.wikimedia.org/wiki/File:Rubicks_cube_IMG_5429.jpg) | Rama / CC BY-SA 2.0 fr | 12.844 s | 664 | Yes |
| [Fidget spinner](https://commons.wikimedia.org/wiki/File:Fidget_spinner_red,_cropped.jpg) | Árni Dagur / CC0 | 13.509 s | 507 | Yes |
| [Corkscrew](https://commons.wikimedia.org/wiki/File:Cork_Screw_Red.jpg) | Kyle May / CC BY 2.0 | 23.973 s | 345 | No |
| [Needle-nose pliers](https://commons.wikimedia.org/wiki/File:Basic_NeedleNose_Pliers.jpg) | oomlout / CC BY-SA 2.0 | 7.926 s | 457 | No |

The exact source URLs, license URLs, dimensions and SHA-256 hashes are in
`backend/app/recon/web_photo_sources.json`. Original JPEGs are downloaded without
editing; the model internally limits the longest image side to 768 pixels and the
mesh raster to 192. The preparation tool rejects changed source bytes and retains
failed attempts. The five network downloads were reproduced and all checksums
matched. A separate local-reuse preparation also passed.

Reconstruction took 81.485 seconds in this run, including model loading and mesh
export but excluding the following LBM calls. Each LBM call took 1.234–1.384 seconds.
These are local observations, not a performance guarantee. Solver conditions were
128×64, 600 steps, tau 0.6, inlet velocity 0.08 lattice units, wind control 15 m/s,
and inlet angle 0°. Normalized geometry has no measured physical scale. Existing
Cd/Cl outputs are educational heuristics/proxies, not measured coefficients.

## Correction exposed by real photographs

The old depth mesh builder independently mapped both image axes into [-1,1].
This squeezed a landscape car image and widened a portrait corkscrew before the
already proportion-preserving mask fitter ran. The builder now scales both axes
from the same original-image extent, including when a thin mesh raster dimension
is clamped to its minimum size.

| Case | Foreground width/height | Old mesh width/height | Corrected mesh width/height |
| --- | ---: | ---: | ---: |
| Car | about 1.79 | 1.09 | 1.81 |
| Corkscrew | about 0.53 | 0.79 | 0.53 |

All five corrected mesh ratios were within about 1% of their extracted foreground
ratios; this checks coordinate consistency with the heuristic mask, not the true
object geometry. Four regression cases cover landscape, portrait, square and a
very thin image. The imported-mask preview also now reports its actual grid size
before simulation instead of saying “Procedural preview.”

## Visual inspection and remaining limits

- **Car:** recognizable relief, but the foreground segmentation includes its cast
  shadow. The silhouette therefore contains background geometry.
- **Cube:** recognizable relief. Its center-depth slice is triangular, which is
  a cross-section of the approximate relief rather than the full photographed outline.
- **Spinner:** recognizable relief and a useful three-lobed slice with holes.
  The bearings are shallow relative-depth features, not reconstructed mechanical parts.
- **Corkscrew:** the pale metal handle/upper stem is missed by foreground extraction;
  thin details are simplified and the serialized mesh is not watertight.
- **Pliers:** relief preserves overall proportions, but the center slice drops the
  elevated jaws and separates the handles; the serialized mesh is not watertight.

The existing mesh-validity flag checks finite vertices and nondegenerate faces;
watertightness is recorded separately. Consequently, 5/5 completed executions does
not mean 5/5 closed meshes. Only 3/5 were watertight. No ground-truth 3D alignment,
geometry accuracy score or CFD calibration was performed. Thin reflective objects
and shadow segmentation need further work; a GPU alone does not resolve these limits.

The first evaluation is preserved as `evaluation_outputs/web_photo_cpu_20261013_attempt1/`.
It includes the pre-correction distortion and a rejected two-tool hammer photograph.
The clean corrected run is `evaluation_outputs/web_photo_cpu_20261013_attempt2/`.
Its `pipeline-inspection.png` compares each photo, depth relief, mask and LBM field.
`inspection.json` records proportion checks and mesh closure without hiding failures.

## Browser verification

The photographed spinner's exported mask was imported into the browser, inspected,
and submitted to the real CPU simulation endpoint. The UI displayed a live 2D LBM
field rather than a cached demo. The active desktop viewer reported 113–120 FPS
while idle and 76 FPS in one sample during CPU inference. These are sampled UI
estimates, not a continuous frame-time benchmark. The earlier 1–6 FPS automation
observations did not reproduce in this visible session, so no speculative rendering
optimization was applied. Physical-phone performance remains untested.
The imported case also fits a 390×844 emulated viewport with document width 390.
Pause/play states were exercised; pixel comparisons were not exact because of
minor camera/render settling, so no bit-for-bit pause proof is claimed.

The CPU baseline and surrogate weights are retained. All 42 backend tests, frontend
production build and lint passed. Build retains the existing large-bundle advisory;
the test client emits an upstream deprecation warning.

## Reproduce and hand off

Run from the repository root in the configured CPU reconstruction environment:

```powershell
python backend/scripts/prepare_web_photo_manifest.py --output-dir evaluation_outputs/web_photos
python backend/scripts/verify_recon_cpu.py evaluation_outputs/web_photos/manifest.csv --engine depth_anything --output-dir evaluation_outputs/web_photo_cpu_attempt1
```

Choose fresh output directories for every attempt. `--source-dir` can reuse local
originals while verifying hashes. `--ca-file` accepts a trusted CA bundle for a
network with a local certificate authority; certificate verification remains enabled.

The GPU collaborator can use the same manifest with the following
reconstruction-only command, then SF3D if the hardware/runtime
passes the current 6144 MiB selection gate:

```powershell
python backend/scripts/evaluate_recon.py evaluation_outputs/web_photos/manifest.csv --engine depth_anything --device cuda --output-dir evaluation_outputs/web_photo_cuda_attempt1
```

Record the selected engine, device, elapsed time, peak VRAM, mesh validity, masks and
failed attempts. Compare geometry and execution with this CPU baseline. Retain
the source/author/license records whenever sharing reproduced imagery.
