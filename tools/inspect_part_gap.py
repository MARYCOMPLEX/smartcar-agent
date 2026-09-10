"""Locate measured closest-point witnesses without changing generated geometry."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh
from smartcar.geometry.solid import from_mesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--a',default='body');p.add_argument('--b',default='bottom_cover');args=p.parse_args()
    meshes={name:trimesh.load(args.run/f'09_structure/{name}.stl',force='mesh') for name in [args.a,args.b]}
    a,b=[meshes[name] for name in [args.a,args.b]]
    print('solid_gap_mm',from_mesh(a).min_gap(from_mesh(b),1),flush=True)
    best=[]
    for source,target in [(args.a,args.b),(args.b,args.a)]:
        points=meshes[source].vertices
        for start in range(0,len(points),1024):
            sample=points[start:start+1024]
            closest,distance,face=trimesh.proximity.closest_point(meshes[target],sample)
            for j in np.argsort(distance)[:4]:
                best.append(dict(source=source,target=target,distance_mm=float(distance[j]),source_point=sample[j].tolist(),target_point=closest[j].tolist(),target_face=int(face[j])))
        best=sorted(best,key=lambda x:x['distance_mm'])[:16]
    print(json.dumps(best,indent=2),flush=True)


if __name__=='__main__':main()
