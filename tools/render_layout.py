"""Read-only inspection of an actual pipeline layout, including failed runs."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from smartcar.io import read_json
from smartcar.render import render_meshes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--iteration',help='Design iteration directory name; defaults to the latest saved layout.')
    args=parser.parse_args();run=args.run.resolve()
    if args.iteration:
        iteration=run/'design_iterations'/args.iteration
    else:
        candidates=sorted(p.parent.parent for p in (run/'design_iterations').glob('*/08_layout/layout.json'))
        if not candidates:parser.error('This run has no saved design iteration layout.')
        iteration=candidates[-1]
    layout=read_json(iteration/'08_layout/layout.json')
    frame=read_json(run/'03_coordinate_frame/frame.json')
    design=read_json(iteration/'03_coordinate_frame/design_frame.json')
    body=trimesh.load(run/'02_repaired/envelope.stl',force='mesh')
    body.apply_transform(np.asarray(design['input_to_design'])@np.asarray(frame['vehicle_to_input']))
    meshes=[('body',body)]
    for inst in layout:
        mesh=trimesh.load(run/'01_input'/f"{inst['role']}.stl",force='mesh')
        meshes.append((inst['id'],mesh.apply_transform(inst['pose']['matrix'])))
    out=iteration/'diagnostics';out.mkdir(exist_ok=True)
    target=out/'layout_inspection.png'
    render_meshes(target,meshes,'Saved solver layout - validation pending or see run report',transparent=True)
    print(target)


if __name__=='__main__':main()
