"""Read-only opposite-surface witnesses for supplementary wall audits."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh
from smartcar.io import read_json

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
for row in read_json(args.audit)['models']:
    if row['additional_status']!='FAIL':continue
    mesh=trimesh.load(root/'runs'/row['run']/'output/body.stl',force='mesh')
    points=np.asarray(row['measurements']['examples'])
    close,d,faces=trimesh.proximity.closest_point(mesh,points);normal=mesh.face_normals[faces]
    origins=close-normal*.02
    hits,rays,triangles=mesh.ray.intersects_location(origins,-normal,multiple_hits=True)
    distances=np.linalg.norm(hits-origins[rays],axis=1)+.02
    witnesses=[]
    for i in range(len(points)):
        ids=np.flatnonzero((rays==i)&(distances>.04))
        if not len(ids):continue
        k=ids[np.argmin(distances[ids])]
        witnesses.append(dict(point=close[i].tolist(),distance=float(distances[k]),
                              normal_dot=float(normal[i]@mesh.face_normals[triangles[k]])))
    print(json.dumps(dict(model=row['model'],witnesses=witnesses[:6]),ensure_ascii=False),flush=True)
