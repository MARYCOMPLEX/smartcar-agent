"""Read-only attribution of strap-channel obstructions to print parts."""
import argparse
from pathlib import Path
import trimesh
from smartcar.io import read_json
from smartcar.geometry.solid import box,from_mesh,to_mesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    records=read_json(args.run/'09_structure/structure.json')
    for name in ['body','bottom_cover','drive_left_retainer','drive_right_retainer']:
        mesh=trimesh.load(args.run/f'09_structure/{name}.stl',force='mesh');solid=from_mesh(mesh)
        for rec in records['battery']:
            for index,b in enumerate(rec['strap_passage_bounds']):
                overlap=solid^box(b)
                print(name,index,overlap.volume(),to_mesh(overlap).bounds.tolist() if not overlap.is_empty() else None,flush=True)


if __name__=='__main__':main()
