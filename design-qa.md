# AeroVoxel option 2 design QA

Visual target: second displayed generated concept (Open Tunnel). Desktop 1440 x 1024; mobile 390 x 844.

First comparison: blocked. Fixed blank renderer caused by invalid particle indexing, low geometry visibility, inaccessible mobile upload label, and fixed-width mobile layout.

Required surfaces: Outfit/monospace typography retained; horizontal object picker, large canvas, compact bottom dock, and collapsible right results match selected hierarchy. Navy/cyan tokens and readable secondary text retained. Scientific geometry and field traces are live WebGL output, intentionally not raster illustrations or fabricated CFD. Relative pressure labels replace unsupported calibrated Cp labels. Actual simulated dimensions remain accurate. Library icons represent objects; no decorative raster assets are required. Original API/upload/surrogate behavior is retained.

Verified interactions: object selection updates estimates; perspective/side switching; pause/play state; results collapse/expand. Mobile reflow has document width 390 at viewport width 390, controls below canvas, results below controls. Reduced-motion CSS and initial paused state are implemented; physical-device and assistive-technology testing not performed.

Evidence: local screenshots under the chat visualization frontend-audit directory, including the source/implementation comparison (11-design-comparison.png), desktop capture and mobile capture. GPU/neural model runtime is outside visual QA.

Follow-up polish: generated concept has decorative wake swirls and a smooth reconstructed object; actual prototype uses its honest educational geometry and actual loaded fields.

final result: passed

Final connected verification: imported a real SYNTH_03 Depth Anything mask through the browser, inspected it, ran CPU LBM, and observed live results. The existing surrogate returned a prediction. New source/parameter hints explicitly distinguish cached fields. Advanced wake and voxel switches were exercised. Fresh browser logs contained no errors. Final mobile document width remained 390 at width 390. Latest accepted captures: 14-landing-final.png, 15-connected-desktop-final.png, 16-connected-mobile-final.png. Real fields retain coarse grid geometry; the generated mockup's smooth geometry and stylized wake are intentionally not represented as computed CFD.


## Airflow motion refinement

User feedback: bright fixed paths with moving beads looked like a static diagram.
Replaced beads with soft, fading camera-facing trails advected through bilinearly
sampled fluid velocity. Midpoint integration and solid-cell checks keep tracers
outside the obstacle. Static guides are now faint. Tracers retain motion state
when changing pressure/geometry display; pause freezes age, advection and trail
history while camera controls remain active. Offline preview remains explicitly
illustrative and uses a separate potential-flow field.

Build and lint passed. Browser shader/error log was empty. A paused canvas crop
(200,100)-(980,550) was pixel-identical across two captures; camera/UI overlays
were excluded from that comparison. Captures 18/19 show motion; 20/21 preserve
the pause check; 22 records the revised live preview. Observed scene reported
60 FPS locally; no general performance guarantee is implied.

final result: passed


## Recognizable demo geometry refinement

| Before | After | Why |
| --- | --- | --- |
| Every loaded case displayed as cell blocks | Car with body/cabin/wheels/lights, four-rotor drone, parametric NACA wing, smooth cylinder and sphere | Named demo figures should be recognizable |
| X and Y lattice cells displayed at different scales | Equal cell size on both axes, shared by masks and velocity lookup | Avoid visually oval circles and distorted airfoil thickness |
| Camera target fixed at domain origin | Model-aware framing, including narrow-screen fit | Keep upstream and large objects visible |

These meshes are native Three.js demo models, not reconstructed CAD or new CFD
geometry. The existing mask remains the obstacle used by the solver. Voxel view
still exposes it, and the display mode is identified in the footer/details.
Custom/imported masks retain their original geometry, without inferred vehicle
classification. The NACA 0012 model uses the original finite-trailing-edge
four-digit equation documented by NASA TMR and used in the cache generator:
https://tmbwg.github.io/turbmodels/naca0012_val.html

Verification: production build and ESLint passed; all five generated models
have finite geometry, sphere bounds are equal on all axes, square masks retain
square X/Y bounds, and sampled NACA thickness/chord is approximately 0.12.
Browser checks covered all named demos, perspective/side, voxel restoration,
and 390x844 mobile fit (document width 390). Browser error log was empty.
Accepted captures: 23-smooth-car.png through 29-smooth-car-mobile.png.

final result: passed

## Detailed geometry, materials and matching flow refinement

The previous procedural car still looked like a toy, with simple material response
and a display silhouette different from the old cached obstacle. The licensed local
CarConcept asset replaces it. Panel lines, wheels, lights and glass now read as a
vehicle. Reflections, a dark continuous floor, restrained shadows and camera fit
give the object a clear place in the scene. The other named models retain their
recognizable shapes and the selected Open Tunnel layout.

The source/implementation comparison (42-final-concept-comparison.png) was inspected.
Header, horizontal library, large tunnel, cyan controls, typography and collapsible
results preserve the approved hierarchy. User feedback after that concept explicitly
requested moving air; the current smoke treatment follows that feedback. Loaded
fields are computed on the 2D grid, rather than decorative wake spirals from the
generated concept. Geometry and field scale share the same lattice mapping.

Concrete corrections during this pass:

| Finding | Correction | Evidence |
| --- | --- | --- |
| Simple car silhouette and flat materials | Detailed licensed model, tuned paint/glass/rubber, environment reflections | 38-final-car-matched-flow.png |
| Mottled paint and glowing wheels in first derivative | Preserve texture UV transforms; remove emissive maps and their flat emissive factors; reduce paint normal intensity | 32/33 preserve the failure and correction; 38 is final |
| Old obstacle data remained in browser cache | Revalidate array requests and check running API array hashes | 36 preserves the rejected stale mask; 37-refreshed-car-mask.png shows the correct obstacle |
| Display model and cached field represented different shapes | Project actual drawn triangles to shared car/drone masks and regenerate CPU fields | Geometry comparison and three field-contract tests passed |
| Hard rectangular smoke tips and strong lighting | Smooth ribbon head/tail opacity, narrower ribbons, subdued lighting and continuous dark floor | 38 and 43 |

Verification: build and lint passed; full backend suite 36/36; projection comparison
passed. Side/perspective, model loading, voxel toggle, pause/play and responsive
layout were exercised. Mobile 390x844 document width is 390; car, legend and controls
fit in capture 43-final-car-mobile-verified.png. Current production browser errors
were empty; historical development-server errors remain in the browser log.

Visual acceptance does not establish CFD accuracy or a frame-rate target. The field
remains coarse 2D CPU LBM. The local car asset is approximately 11.5 MB. Automated
captures lagged some viewport/state changes and FPS samples were low even for basic
scenes; physical-device animation performance and network loading are unmeasured.
Final airfoil section and quadcopter captures are 48 and 49; the final car capture
is 50. Pause/play states were verified in the UI, and the renderer freezes marker
age, integration and history while paused. Pixel comparisons (39/40 and 44/45)
were not identical because the camera was still settling after view/viewport
changes; these attempts remain preserved and are not claimed as pixel-exact proof.

final result: passed (visual layout, geometry, materials and field alignment)

## Real-photo browser follow-up

Imported the real photographed spinner's voxel slice, confirmed its actual
128x64 preview grid, and ran the existing CPU simulation from the browser.
Capture 51 shows the computed field; capture 55 shows the 390x844 mobile layout
(document width 390). The visible desktop viewer reported 113-120 FPS at idle and
76 FPS in one sample during CPU inference; timing samples are preserved in
web-photo-performance.json. Earlier low automation FPS did not reproduce here.
No speculative renderer optimization or visual redesign was applied. Physical
phone performance remains untested. Pause states were exercised but pixel
comparisons 52/53 were not exact; raw attempts remain preserved.
