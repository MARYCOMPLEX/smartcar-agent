"""Read-only strengthened wall audit of immutable completed designs."""
import argparse
import datetime
from pathlib import Path
import numpy as np
import trimesh
from smartcar.io import read_json,write_json,sha256
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import to_mesh
from smartcar.structure.closure_ribs import rib_solid
from smartcar.validation.geometry import wall_measurements


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--indices',type=int,nargs='+');args=p.parse_args()
    root=Path(__file__).resolve().parents[1];suite=args.suite.resolve();manifest=read_json(suite/'manifest.json')
    known={m['index'] for m in manifest['models']}
    if args.indices is not None and (not args.indices or len(set(args.indices))!=len(args.indices) or set(args.indices)-known):
        p.error('Indices must be distinct known model indices.')
    rows=[];missing=[]
    for model in manifest['models']:
        if args.indices and model['index'] not in args.indices:continue
        run=root/f'runs/{suite.name}-{args.label}-m{model["index"]:02d}'
        if not (run/'output/body.stl').exists():
            missing.append(dict(index=model['index'],model=model['source_entry'],run=run.name));continue
        profile=ManufacturingProfile(read_json(run/'01_input/provenance.json')['manufacturing'])
        records=read_json(run/'09_structure/structure.json');regions=[]
        for rib in records['closure'].get('connection_ribs',[]):
            bounds=to_mesh(rib_solid(rib)).bounds.copy()
            bounds+=np.array([[-1]*3,[1]*3])*profile.nominal_wall;regions.append(bounds)
        mesh=trimesh.load(run/'output/body.stl',force='mesh')
        measured=wall_measurements(mesh,profile,regions=regions)
        report=read_json(run/'output/validation_report.json')
        row=dict(index=model['index'],model=model['source_entry'],run=run.name,
            original_report_sha256=sha256(run/'output/validation_report.json'),original_counts=report['counts'],
            additional_status='FAIL' if measured['below_minimum'] else 'WARNING',required_mm=profile.minimum_wall,measurements=measured)
        rows.append(row);print(model['source_entry']+' '+row['additional_status']+f" min={measured['minimum_mm']:.6f} below={measured['below_minimum']} rays={measured['samples']}",flush=True)
    output=suite/'reports'/f'{args.label}-critical-wall-audit.json'
    write_json(output,dict(kind='separate read-only audit; original geometry and engineering reports are unchanged',
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        corpus_model_count=len(manifest['models']),requested_indices=args.indices,audited_model_count=len(rows),missing_requested_models=missing,
        validator_source_sha256=sha256(root/'src/smartcar/validation/geometry.py'),cohort=args.label,models=rows))
    print(str(output),flush=True)


if __name__=='__main__':main()
