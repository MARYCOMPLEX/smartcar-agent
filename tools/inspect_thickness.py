"""Read-only inspection of the independent validator's shortest wall rays."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh

p=argparse.ArgumentParser();p.add_argument('output',type=Path)
p.add_argument('--reference-report',type=Path,help='Recheck previous measured witness locations against a newer mesh without editing either run.')
p.add_argument('--output-json',type=Path,help='Save a separate read-only diagnostic record.')
args=p.parse_args()
m=trimesh.load(args.output/'body.stl',force='mesh')
report=json.loads((args.reference_report or args.output/'validation_report.json').read_text())
q=next(c for c in report['checks'] if c['check']=='wall_thickness:body')['measurements']['examples']
close,d,f=trimesh.proximity.closest_point(m,np.array(q));n=m.face_normals[f]
hits,rays,t=m.ray.intersects_location(close-n*.02,-n,multiple_hits=True)
dist=np.linalg.norm(hits-(close-n*.02)[rays],axis=1)+.02
rows=[]
for i in range(len(q)):
    eligible=np.flatnonzero((rays==i)&(dist>.04))
    if not len(eligible):
        rows.append(dict(point=close[i].tolist(),resolved=False));continue
    k=eligible[np.argmin(dist[eligible])]
    rows.append(dict(point=np.round(close[i],3).tolist(),normal=np.round(n[i],3).tolist(),distance=round(float(dist[k]),4),
                     source_surface_distance_mm=float(d[i]),
                     target=np.round(hits[k],3).tolist(),opposite_normal=np.round(m.face_normals[t[k]],3).tolist(),normal_dot=round(float(n[i]@m.face_normals[t[k]]),3)))
print(json.dumps(rows,indent=2))
if args.output_json:
    args.output_json.parent.mkdir(exist_ok=True,parents=True)
    args.output_json.write_text(json.dumps(dict(kind='read-only witness recheck',mesh=str(args.output/'body.stl'),
        reference_report=str(args.reference_report or args.output/'validation_report.json'),measurements=rows),indent=2),encoding='utf-8')
