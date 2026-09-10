# Code map

| Module | Responsibility |
|---|---|
| `src/smartcar/pipeline.py` | CLI, immutable run folders, input provenance, geometry analysis |
| `batch.py` | ZIP inventory, pinned source/profile cohorts, per-model public CLI runs and timing summaries |
| `domain/vehicle_policy.py`, `config/vehicle_design.json` | Explicit axle proportions, source correspondence and local appearance-redesign policy; separate from physical manufacturing tolerances |
| `domain/hardware.py` | STEP import, intrinsic rigid bounds, CAD shaft/holes/planes, support-column certificates |
| `domain/assembly.py` | Hardware instances, poses, conservative proxies with certified empty mounting regions |
| `geometry/repair.py`, `errors.py` | Pre-allocation grid budget, explicit empty-volume diagnostics, voxel closing/fill and robust appearance envelope |
| `geometry/sdf.py` | Distance field, physical-resolution resampling after scale changes, separable full-box erosion |
| `understanding/coordinate_frame.py`, `wheel_evidence.py` | All OBB/PCA axis permutations, bilateral symmetry, ground/paired-wheel/connected-axle evidence, ambiguous front heuristic |
| `understanding/input_scale.py` | Explicit source-unit or normalized Y-length calibration, immutable source geometry, audited source-to-mm transforms |
| `layout/wheel_candidates.py` | Fixed motor-wheel modules with horizontal/upright rolls, measured wheel candidates and sequential hard-filter diagnostics |
| `understanding/axle_anchors.py`, `ground_frame.py` | Two bilateral low contact bands and direct source-wheel evidence; fused-tire frame recovery |
| `geometry/wheelwell_envelope.py`, `vehicle_section.py` | Bounded wheel-region redesign from measured crown/flank/belly surfaces and axle-local body widths |
| `layout/axle_solver.py`, `validation/axle_layout.py` | Joint front/rear search and independent pose-based wheelbase, overhang, source-arch and track checks |
| `structure/support_connections.py`, `bottom_regularization.py` | Bottom-plane edge cleanup; rails stay within bottom thickness, measure overlap or shared contact area, and avoid actual access/wheel cuts |
| `structure/closure.py` | Screw-site search checks full native bosses against generated chassis/support clearance before selecting four closure locations |
| `structure/lightweight.py`, `closure_ribs.py` | Closure connections to the actual cut shell; preserved native ribs/pillars survive shell reconstruction |
| `layout/hardware_candidates.py`, `solver.py` | Discrete orientations, candidate filtering, CP-SAT hard conflicts |
| `structure/` | Cavity, PCB posts, motor cradle/retainers, battery tray, openings and screw closure |
| `structure/regularize.py`, `junction_cleanup.py`, `closure_bores.py` | Bounded shell reconstruction, local junction material, native seam and pilots derived from the actual screw bearing datum |
| `geometry/print_mesh.py` | Preserve final surfaces; condition the original double-precision solid only when measured float32 STL topology or tiny-area faces fail, retaining the physical volume-change gate |
| `assembly/` | Insertion order, sampled collision checks, conservative continuous certificates |
| `validation/` | Independent measured checks, no report edits or relaxed thresholds |
| `agent/planner.py` | Strategy queue, failure history and scale exploration |
| `output.py`, `render.py` | STL/GLB/3MF, plots, BOM and design report |

Geometry owns all coordinates and distances. Policy selects strategies only. Tests exercise geometric counterexamples, CAD intrinsic recovery and manufacturing regressions.

`tools/summarize_archive.py`, `audit_delivery.py`, `audit_critical_walls.py` and `render_wall_evidence.py` are reporting/read-only diagnostics. They never alter a run's STL, hardware poses or original validation report. A positive sampled wall result is not a global wall-thickness proof.
