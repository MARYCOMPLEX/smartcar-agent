"""Create a synthetic rotated geometry input, then use the FORMAL pipeline.

This does not generate designs or alter output files. Dimensions describe an
independent analytic test enclosure, not a rule for any uploaded vehicle.
"""
from pathlib import Path
import subprocess
import sys
import trimesh

root=Path(__file__).resolve().parents[1]
directory=root/"benchmarks/inputs";directory.mkdir(parents=True,exist_ok=True)
path=directory/"rotated_analytic_enclosure.stl"
mesh=trimesh.creation.box([90,210,65])
mesh.apply_transform(trimesh.transformations.rotation_matrix(.61,[1,2,3]))
mesh.apply_translation([37,-19,28]);mesh.export(path)
subprocess.run([sys.executable,"-m","smartcar.pipeline","--appearance",str(path),"--run-id","benchmark-rotated-box"],cwd=root,check=True)
