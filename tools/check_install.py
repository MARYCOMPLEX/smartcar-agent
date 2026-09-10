"""Check the cloned inputs and native runtime without using any historical cache."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("source_bundle/smartcar-agent-foundation-v1/data")


def check_inputs(root: Path) -> list[dict]:
    """Resolve all kit references relative to their definitions and verify bytes."""
    records = []
    base = root / DATA
    appearance = json.loads((base / "appearance/appearance.json").read_text(encoding="utf-8"))
    references = [(base / "appearance", appearance["geometry"])]
    manifest = json.loads((base / "hardware/kit_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["components"]:
        definition = base / "hardware" / item["path"]
        raw = json.loads(definition.read_text(encoding="utf-8"))
        geometry = raw["geometry"]
        references.append((definition.parent, geometry if "path" in geometry else geometry["collision_proxy"]))
    for directory, geometry in references:
        path = (directory / geometry["path"]).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError(f"Input reference leaves bundled data: {geometry['path']}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = geometry.get("sha256")
        if not expected or actual != expected:
            raise ValueError(f"Bundled input checksum mismatch: {path.relative_to(root)}")
        records.append(dict(path=path.relative_to(root).as_posix(), sha256=actual, bytes=path.stat().st_size))
    for name in ("manufacturing.json", "vehicle_design.json"):
        json.loads((root / "config" / name).read_text(encoding="utf-8"))
    return records


NATIVE_CHECK = r'''
import tempfile
from pathlib import Path
import cadquery as cq
import manifold3d as m3d
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from ortools.sat.python import cp_model
import scipy.ndimage
import skimage.measure
import pymeshfix
import fast_simplification
import lxml.etree
import xxhash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / 'roundtrip.step'
    cube = cq.Workplane('XY').box(2, 3, 4)
    cq.exporters.export(cube, str(target))
    restored = cq.importers.importStep(str(target)).val()
    assert abs(restored.Volume() - 24) < 1e-6
    assert abs(m3d.Manifold.cube([2, 3, 4]).volume() - 24) < 1e-6
    intersector = RayMeshIntersector(trimesh.creation.box([2, 3, 4]))
    hits = intersector.intersects_any(np.array([[0., 0., 10.]]), np.array([[0., 0., -1.]]))
    assert bool(hits[0])
    model = cp_model.CpModel()
    value = model.new_int_var(0, 1, 'value')
    model.add(value == 1)
    solver = cp_model.CpSolver()
    assert solver.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    plt.plot([0, 1], [0, 1])
    plt.savefig(Path(directory) / 'render.png')
    plt.close()
print('NATIVE_RUNTIME_PASS')
'''


def main() -> None:
    started = time.perf_counter()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12; run setup.cmd to prepare the tested environment.")
    records = check_inputs(ROOT)
    result = subprocess.run([sys.executable, "-c", NATIVE_CHECK], cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(f"Native geometry check failed with exit code {result.returncode}.")
    report = dict(status="PASS", python=platform.python_version(), platform=platform.platform(),
                  input_assets=records, seconds=round(time.perf_counter() - started, 3))
    output = ROOT / "reports/install_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"INSTALL_CHECK_PASS: {len(records)} original geometry assets, CAD roundtrip, collision, solver and render.")
    print(f"Report: {output}")
    print("Run the included car: run.cmd")
    print('Run another STL: run.cmd --appearance "C:/models/car.stl"')


if __name__ == "__main__":
    main()
