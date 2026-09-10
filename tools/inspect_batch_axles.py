"""Read-only diagnosis of measured axle evidence from formal pipeline inputs."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from smartcar.io import read_json, write_json
from smartcar.understanding.axle_anchors import detect_axle_anchors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',type=Path,required=True)
    args=parser.parse_args();suite=args.suite.resolve();root=Path(__file__).resolve().parents[1]
    destination=suite/'diagnostics';destination.mkdir(exist_ok=True)
    records=[];fig,axes=plt.subplots(5,2,figsize=(16,17))
    for model,ax in zip(read_json(suite/'manifest.json')['models'],axes.flat):
        i=model['index'];choices=[root/f'runs/{suite.name}-analysis-size200-m{i:02d}',root/f'runs/{suite.name}-baseline-v2-m{i:02d}']
        run=next((r for r in choices if (r/'03_coordinate_frame/normalized.stl').exists()),None)
        if run is None:
            ax.set_title(model['source_entry']+' : analysis pending');continue
        mesh=trimesh.load(run/'03_coordinate_frame/normalized.stl',force='mesh')
        semantic=read_json(run/'03_coordinate_frame/semantic_regions.json') if (run/'03_coordinate_frame/semantic_regions.json').exists() else {}
        evidence=detect_axle_anchors(mesh,semantic)
        records.append(dict(model=model['source_entry'],run=str(run),evidence=evidence))
        faces=mesh.triangles[::max(1,len(mesh.faces)//18000)]
        ax.add_collection(PolyCollection(faces[:,:,1:],facecolor='#b6c9d2',edgecolor='none',alpha=.4))
        for axle in evidence['axles']:
            ax.axvline(axle['y_mm'],color='#d34823',lw=2)
            if axle.get('radius_mm') and axle.get('z_mm'):
                from matplotlib.patches import Circle
                ax.add_patch(Circle((axle['y_mm'],axle['z_mm']),axle['radius_mm'],fill=False,color='#d34823',lw=1))
        ax.set_xlim(mesh.bounds[:,1]+np.array([-1,1])*mesh.extents[1]*.03)
        ax.set_ylim(mesh.bounds[0,2]-mesh.extents[2]*.05,mesh.bounds[1,2]+mesh.extents[2]*.07)
        ax.set_aspect('equal');ax.set_title(model['source_entry']+' | '+evidence['confidence']+f" | L={mesh.extents[1]:.1f} mm")
        ax.set_xlabel('Y / mm');ax.set_ylabel('Z / mm');ax.grid(alpha=.2)
        print(model['source_entry'],evidence['method'],[(round(a['y_mm'],2),round(a['radius_mm'],2) if a['radius_mm'] else None) for a in evidence['axles']],flush=True)
    fig.tight_layout();fig.savefig(destination/'source_axle_contact_sheet.png',dpi=120);plt.close(fig)
    write_json(destination/'source_axle_evidence.json',records)


if __name__=='__main__':main()
