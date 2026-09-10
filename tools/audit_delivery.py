"""Read-only export/provenance audit; never changes a pipeline acceptance report."""
import argparse
import collections
import datetime
import time
from array import array
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET
import numpy as np
import trimesh
from smartcar.io import read_json, write_json, sha256
from smartcar.geometry.mesh import mesh_stats


def read_3mf_parts(stream):
    """Stream large print meshes without retaining millions of XML elements."""
    stack=[];vertices=array('d');faces=array('q');name=None
    for event,node in ET.iterparse(stream,events=('start','end')):
        local=node.tag.rsplit('}',1)[-1]
        if event=='start':
            stack.append(node)
            if local=='model':yield 'unit',node.attrib.get('unit')
            elif local=='object':name=node.attrib.get('name');vertices=array('d');faces=array('q')
            continue
        if local=='vertex':
            vertices.extend(float(node.attrib[k]) for k in ('x','y','z'));node.clear()
            if len(vertices)%(3*8192)==0:stack[-2].clear()
        elif local=='triangle':
            faces.extend(int(node.attrib[k]) for k in ('v1','v2','v3'));node.clear()
            if len(faces)%(3*8192)==0:stack[-2].clear()
        elif local=='object':
            yield name,trimesh.Trimesh(np.frombuffer(vertices,dtype=np.float64).reshape(-1,3),
                np.frombuffer(faces,dtype=np.int64).reshape(-1,3),process=True)
            node.clear()
        stack.pop()


def audit_run(run, model, suite, implementation, root, critical_walls=False):
    output=run/'output'; checks=[]
    def add(name, passed, **data):
        checks.append(dict(check=name,status='PASS' if passed else 'FAIL',measurements=data))
    required=['body.stl','bottom_cover.stl','assembly.glb','exploded_assembly.glb',
              'layout.json','assembly_plan.json','validation_report.json','bom.json','design_report.md','printable.3mf']
    missing=[name for name in required if not (output/name).is_file()]
    add('required_artifacts',not missing,missing=missing)
    if missing:return dict(run=run.name,model=model['source_entry'],checks=checks,status='INCOMPLETE')
    run_state=read_json(run/'run_status.json').get('status')
    add('completed_run',run_state in ('WARNING','FAIL'),run_status=run_state)
    add('source_preserved',sha256(suite/model['path'])==sha256(run/'01_input/appearance.stl')==model['sha256'])
    snapshot=run/'01_input/source_snapshot'
    actual={str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*.py')}
    add('frozen_source',actual==implementation['source_sha256'],python_files=len(actual))
    current={str(p.relative_to(root/'src')):sha256(p) for p in (root/'src').rglob('*.py')}
    add('current_source_matches_tested',current==actual)
    provenance=read_json(run/'01_input/provenance.json')
    add('manufacturing_profile_preserved',provenance['manufacturing']==read_json(root/'config/manufacturing.json'))
    add('vehicle_policy_preserved',read_json(run/'01_input/vehicle_policy.json')==read_json(root/'config/vehicle_design.json'))
    report=read_json(output/'validation_report.json'); counts=dict(collections.Counter(c['status'] for c in report['checks']))
    add('report_counts_consistent',all(report['counts'][k]==counts.get(k,0) for k in ['PASS','FAIL','WARNING']),counts=report['counts'])
    add('no_formal_failures',report['counts']['FAIL']==0,blocking_checks=report['blocking_checks'])
    add('unresolved_release_remains_explicit',report['release_ready'] is False)
    bom=read_json(output/'bom.json'); originals={}; stats={}; wall_audit=None
    for name in bom['print_parts']:
        mesh=trimesh.load(output/(name+'.stl'),force='mesh')
        s=mesh_stats(mesh);s.pop('components',None);stats[name]=s
        add('stl:'+name,s['watertight'] and s['manifold'] and s['winding_consistent'] and s['component_count']==1 and s['degenerate_triangles']==0,**s)
        if critical_walls and name=='body':
            from smartcar.domain.manufacturing import ManufacturingProfile
            from smartcar.geometry.solid import to_mesh
            from smartcar.structure.closure_ribs import rib_solid
            from smartcar.validation.geometry import wall_measurements
            profile=ManufacturingProfile(provenance['manufacturing'])
            regions=[]
            for rib in read_json(run/'09_structure/structure.json')['closure'].get('connection_ribs',[]):
                bounds=to_mesh(rib_solid(rib)).bounds.copy()
                bounds+=np.array([[-1]*3,[1]*3])*profile.nominal_wall
                regions.append(bounds)
            measured=wall_measurements(mesh,profile,regions=regions)
            wall_audit=dict(index=model['index'],model=model['source_entry'],run=run.name,
                original_report_sha256=sha256(output/'validation_report.json'),original_counts=report['counts'],
                additional_status='FAIL' if measured['below_minimum'] else 'WARNING',
                required_mm=profile.minimum_wall,measurements=measured)
        originals[name]=dict(extents=mesh.extents.copy(),volume=float(mesh.volume))
        del mesh
    part_names=[]
    with zipfile.ZipFile(output/'printable.3mf') as archive:
        add('3mf_crc',archive.testzip() is None)
        with archive.open('3D/3dmodel.model') as stream:
            for name,m in read_3mf_parts(stream):
                if name=='unit':add('3mf_units',m=='millimeter',unit=m);continue
                part_names.append(name);s=mesh_stats(m);s.pop('components',None)
                original=originals.get(name);delta=float(np.max(np.abs(m.extents-original['extents']))) if original is not None else float('inf')
                relative_volume_error=abs(m.volume-original['volume'])/original['volume'] if original is not None else float('inf')
                add('3mf:'+str(name),s['watertight'] and s['manifold'] and s['component_count']==1 and s['winding_consistent'] and s['degenerate_triangles']==0 and delta<=provenance['manufacturing']['numerical_tolerance'] and relative_volume_error<1e-4,
                    extent_error_mm=delta,relative_volume_error=relative_volume_error,**s)
                del m
    add('3mf_part_count',len(part_names)==len(originals) and set(part_names)==set(originals),names=part_names)
    layout=read_json(output/'layout.json');scene=trimesh.load(output/'assembly.glb',force='scene')
    for inst in layout['instances']:
        if inst['id'] not in scene.graph.nodes_geometry:
            add('glb_hardware:'+inst['id'],False,reason='missing node');continue
        transform,geometry=scene.graph[inst['id']]
        m=scene.geometry[geometry].copy().apply_transform(transform).apply_scale(1000.)
        # Authoritative bounds may be a conservative proxy. Only exact wheel
        # proxies are compared directly; all hardware must at least be present.
        error=float(np.max(np.abs(m.bounds-np.asarray(inst['bounding_box']))))
        wheel='wheel' in inst['role']
        add('glb_hardware:'+inst['id'],not wheel or error<=provenance['manufacturing']['numerical_tolerance'],is_wheel=wheel,bounds_difference_mm=error)
    plan=read_json(output/'assembly_plan.json');paths=[s for s in plan['steps'] if 'path' in s]
    expected_paths={i['id'] for i in layout['instances']}|{'body_closure','bottom_lid'}|(set(originals)-{'body','bottom_cover'})
    actual_paths=[s['id'] for s in paths]
    add('modeled_assembly_paths',len(actual_paths)==len(expected_paths) and set(actual_paths)==expected_paths and all(s['path']['status']=='PASS' and s['path'].get('continuous_certificate') for s in paths),path_count=len(paths),expected_paths=sorted(expected_paths),
        paths=[dict(id=s['id'],status=s['path']['status'],continuous_certificate=s['path'].get('continuous_certificate')) for s in paths])
    return dict(run=run.name,model=model['source_entry'],status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
                interpretation='export and provenance audit only; original engineering report and release gates remain authoritative',
                assembly_extents_mm=(scene.extents*1000).tolist(),checks=checks,critical_wall_audit=wall_audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--indices',type=int,nargs='+',help='Audit a completed subset during development.')
    p.add_argument('--critical-walls',action='store_true',help='Also measure final STL walls and save a separate wall-audit record.')
    p.add_argument('--wait-for-completion',action='store_true',help='Audit each immutable completed case while the current foreground batch continues.')
    p.add_argument('--wait-timeout-seconds',type=float,default=14400,help='Bound waiting for missing/running cases; timeouts remain INCOMPLETE.')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];suite=args.suite.resolve();folder=suite/'batches'/args.label
    implementation=read_json(folder/'implementation.json');manifest=read_json(suite/'manifest.json');records=[]
    known={m['index'] for m in manifest['models']}
    if args.indices is not None and (not args.indices or len(set(args.indices))!=len(args.indices) or set(args.indices)-known):
        p.error('Indices must be distinct known model indices; an empty audit is not success.')
    out=suite/'reports';out.mkdir(exist_ok=True)
    archive=Path(manifest['archive'])
    archive_hash=sha256(archive) if archive.exists() else None
    pending=[m for m in manifest['models'] if not args.indices or m['index'] in args.indices]
    wall_rows=[];deadline=time.monotonic()+args.wait_timeout_seconds
    def save():
        remaining=[dict(index=m['index'],model=m['source_entry']) for m in pending]
        write_json(out/f'{args.label}-export-audit.json',dict(kind='separate read-only audit',requested_indices=args.indices,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=not pending,
            pending_models=remaining,corpus_model_count=len(manifest['models']),audited_model_count=len(records),
            original_archive_sha256=archive_hash,expected_archive_sha256=manifest['archive_sha256'],models=records))
        if args.critical_walls:
            audited={r['index'] for r in wall_rows}
            missing=[dict(index=m['index'],model=m['source_entry']) for m in manifest['models']
                if (not args.indices or m['index'] in args.indices) and m['index'] not in audited]
            write_json(out/f'{args.label}-critical-wall-audit.json',dict(
                kind='separate read-only audit; original geometry and engineering reports are unchanged',
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                complete=not missing,corpus_model_count=len(manifest['models']),requested_indices=args.indices,
                audited_model_count=len(wall_rows),missing_requested_models=missing,
                validator_source_sha256=sha256(root/'src/smartcar/validation/geometry.py'),cohort=args.label,models=wall_rows))
    while pending:
        progressed=False
        for model in list(pending):
            run=root/f'runs/{suite.name}-{args.label}-m{model["index"]:02d}'
            status_path=run/'run_status.json'
            if args.wait_for_completion and time.monotonic()<deadline:
                try:status=read_json(status_path).get('status') if status_path.exists() else None
                except (OSError,ValueError):continue  # Writer may be completing the status file.
                if status in (None,'RUNNING'):continue
            try:record=audit_run(run,model,suite,implementation,root,args.critical_walls)
            except Exception as exc:record=dict(run=run.name,model=model['source_entry'],status='ERROR',error=repr(exc))
            wall=record.pop('critical_wall_audit',None)
            if wall is not None:wall_rows.append(wall)
            records.append(record);pending.remove(model);progressed=True;save()
            print(model['source_entry']+' export='+record['status']+
                (f" wall={wall['additional_status']} min={wall['measurements']['minimum_mm']:.6f} mm" if wall else ''),flush=True)
        if pending and not progressed:time.sleep(5)
    save()
    if not records or not all(r['status']=='PASS' for r in records) or any(r['additional_status']=='FAIL' for r in wall_rows):raise SystemExit(1)


if __name__=='__main__':main()
