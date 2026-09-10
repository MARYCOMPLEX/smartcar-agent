"""Read-only local section inspection of generated solids and measured interfaces."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import trimesh
from smartcar.geometry.solid import from_mesh
from smartcar.io import read_json

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
p.add_argument('--z',type=float,required=True);p.add_argument('--xy',type=float,nargs=2,required=True)
p.add_argument('--span',type=float,default=15);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
fig,ax=plt.subplots(figsize=(8,8))
for part,color in [('body','#46758b'),('bottom_cover','#bd8140')]:
    solid=from_mesh(trimesh.load(args.run/f'09_structure/{part}.stl',force='mesh'))
    for polygon in solid.slice(args.z).to_polygons():
        ax.plot(*polygon.T,color=color,linewidth=.8)
record=read_json(args.run/'09_structure/structure.json')
for rib in record['closure'].get('connection_ribs',[]):
    if rib['type']=='vertical':ax.add_patch(Circle(rib['center_xy'],rib['radius_mm'],fill=False,color='#b44551',ls='--'))
x,y=args.xy;ax.plot(x,y,'ro',ms=3);ax.set_xlim(x-args.span/2,x+args.span/2);ax.set_ylim(y-args.span/2,y+args.span/2)
ax.set_aspect('equal');ax.grid(alpha=.3);ax.set_xlabel('X / mm');ax.set_ylabel('Y / mm')
ax.set_title(f'Generated section Z={args.z:g} mm; dashed = native closure rib')
args.output.parent.mkdir(exist_ok=True,parents=True);fig.savefig(args.output,dpi=160)
