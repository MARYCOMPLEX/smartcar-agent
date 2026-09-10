"""Reproducible multi-model testing through the public, immutable pipeline CLI."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import trimesh
from smartcar.io import read_json,write_json,write_csv,sha256

ROOT=Path(__file__).resolve().parents[2]


def prepare(archive,suite):
    if suite.exists():raise ValueError('SUITE_ALREADY_EXISTS')
    suite.mkdir(parents=True);inputs=suite/'inputs';inputs.mkdir()
    records=[];ignored=[]
    with zipfile.ZipFile(archive) as z:
        entries=sorted(z.infolist(),key=lambda x:x.filename)
        for entry in entries:
            relative=PurePosixPath(entry.filename.replace('\\','/'))
            if entry.is_dir():continue
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):raise ValueError('UNSAFE_ARCHIVE_PATH')
            if '__MACOSX' in relative.parts or relative.name.startswith('._'):
                ignored.append(dict(name=entry.filename,reason='macOS metadata, not model geometry'));continue
            if relative.suffix.lower()!='.stl':
                ignored.append(dict(name=entry.filename,reason='unsupported input type; not executed'));continue
            if (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('ARCHIVE_SYMLINK_NOT_ALLOWED')
            target=inputs/Path(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(entry) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            rec=dict(index=len(records)+1,source_entry=entry.filename,path=str(target.relative_to(suite)),bytes=entry.file_size,sha256=sha256(target))
            try:
                mesh=trimesh.load(target,force='mesh')
                rec['source_geometry']=dict(bounds=mesh.bounds.tolist(),extents=mesh.extents.tolist(),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),unit='unspecified by STL')
            except Exception as error:rec['load_error']=repr(error)
            records.append(rec)
    write_json(suite/'manifest.json',dict(archive=str(archive.resolve()),archive_sha256=sha256(archive),models=records,ignored_entries=ignored))
    print(json.dumps(dict(suite=str(suite),models=records,ignored_entries=len(ignored)),ensure_ascii=False),flush=True)


def summarize_run(run):
    result=dict(run=str(run))
    for name in ['run_status.json','scale_search.json']:
        if (run/name).exists():result[name.removesuffix('.json')]=read_json(run/name)
    report=run/'output/validation_report.json'
    if report.exists():
        r=read_json(report);result.update(counts=r['counts'],blocking_checks=r['blocking_checks'],warnings=[c['check'] for c in r['checks'] if c['status']=='WARNING'])
    layout=run/'output/layout.json'
    if layout.exists():
        data=read_json(layout);wheels=[i for i in data['instances'] if 'wheel' in i['role']]
        centers=np.array([np.asarray(i['bounding_box']).mean(0) for i in wheels])
        if len(centers)==4:
            axle_y=sorted(set(round(float(y),5) for y in centers[:,1]))
            result['wheel_metrics']=dict(centers_mm=centers.tolist(),axle_y_mm=axle_y,wheelbase_mm=float(np.ptp(centers[:,1])),track_mm=float(np.ptp(centers[:,0])))
        frame=read_json(run/'03_coordinate_frame/frame.json')
        result['input_dimensions_mm']=frame['dimensions_mm'];result['scale']=data['scale']
        if 'wheel_metrics' in result:
            result['wheel_metrics']['wheelbase_to_input_length']=result['wheel_metrics']['wheelbase_mm']/(frame['dimensions_mm'][1]*data['scale'])
    trace=run/'agent_trace.json'
    if trace.exists():
        events=read_json(trace);result['failure_codes']=sorted(set(str(e.get('error_code',e.get('failure'))) for e in events if e.get('failure')))
    return result


def write_summary(folder,records,manifest,label,stage):
    write_json(folder/'summary.json',dict(label=label,stage=stage,archive_sha256=manifest['archive_sha256'],models=records))
    rows=[]
    for record in sorted(records,key=lambda r:r['index']):
        status=record.get('run_status',{});counts=record.get('counts',{})
        rows.append(dict(model=record['model'],status=status.get('status','NO_STATUS'),exit_code=record.get('exit_code'),seconds=record.get('seconds'),scale=record.get('scale'),
                         PASS=counts.get('PASS'),FAIL=counts.get('FAIL'),WARNING=counts.get('WARNING'),wheelbase_ratio=record.get('wheel_metrics',{}).get('wheelbase_to_input_length'),
                         blockers='; '.join(record.get('blocking_checks',[])),error=status.get('error',''),run=record['run']))
    write_csv(folder/'summary.csv',rows)
    lines=['# Batch pipeline results','',f'Batch: {label}; stage: {stage}. Each model was processed by the public pipeline CLI.',
           'Zero FAIL does not establish sensible axle placement unless the run includes independent axle-layout checks.','',
           '| Model | Status | Scale | PASS / FAIL / WARNING | Wheelbase / length | Seconds |','|---|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['model']} | {r['status']} | {r['scale']} | {r['PASS']} / {r['FAIL']} / {r['WARNING']} | {r['wheelbase_ratio']} | {r['seconds']} |")
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_batch(suite,label,jobs,stage,indices=None,target_length=None,calibration_plan=None):
    manifest=read_json(suite/'manifest.json');folder=suite/'batches'/label
    if folder.exists():raise ValueError('BATCH_ALREADY_EXISTS')
    folder.mkdir(parents=True)
    calibration=read_json(calibration_plan) if calibration_plan else {}
    if calibration_plan:shutil.copy2(calibration_plan,folder/'calibration_plan.json')
    implementation=folder/'runtime/src'
    shutil.copytree(ROOT/'src',implementation,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    shutil.copy2(ROOT/'config/manufacturing.json',folder/'manufacturing.json')
    shutil.copy2(ROOT/'config/vehicle_design.json',folder/'vehicle_design.json')
    environment=os.environ.copy();environment['PYTHONPATH']=str(implementation);environment['SMARTCAR_PROJECT_ROOT']=str(ROOT)
    write_json(folder/'implementation.json',dict(source_sha256={str(p.relative_to(implementation)):sha256(p) for p in implementation.rglob('*.py')},profile_sha256=sha256(folder/'manufacturing.json'),vehicle_policy_sha256=sha256(folder/'vehicle_design.json')))
    by_index={m['index']:m for m in manifest['models']}
    requested=indices if indices is not None else list(by_index)
    if len(requested)!=len(set(requested)):raise ValueError('DUPLICATE_MODEL_INDEX')
    if any(index not in by_index for index in requested):raise ValueError('UNKNOWN_MODEL_INDEX')
    selected=[by_index[index] for index in requested]
    if not selected:raise ValueError('NO_SELECTED_MODELS')
    prefix=re.sub('[^a-zA-Z0-9_-]','-',suite.name)
    def one(model):
        run_id=f'{prefix}-{label}-m{model["index"]:02d}'
        cmd=[sys.executable,'-m','smartcar.pipeline','--appearance',str(suite/model['path']),'--run-id',run_id,'--stage',stage,'--profile',str(folder/'manufacturing.json'),'--vehicle-policy',str(folder/'vehicle_design.json')]
        setting=calibration.get('models',{}).get(str(model['index']),{})
        size=target_length if target_length is not None else setting.get('target_length_mm',model.get('target_length_mm'))
        if size is not None:cmd+=['--target-length-mm',str(size)]
        if model.get('input_unit') and size is None:cmd+=['--input-unit',model['input_unit']]
        start=time.perf_counter();print('BATCH START '+model['source_entry']+' -> '+run_id,flush=True)
        with (folder/f'm{model["index"]:02d}.log').open('w',encoding='utf-8') as log:
            completed=subprocess.run(cmd,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        record=dict(index=model['index'],model=model['source_entry'],input_sha256=model['sha256'],command=cmd,calibration_assumption=setting,exit_code=completed.returncode,seconds=round(time.perf_counter()-start,3),**summarize_run(ROOT/'runs'/run_id))
        write_json(folder/f'm{model["index"]:02d}.json',record)
        print('BATCH END '+model['source_entry']+' '+record.get('run_status',{}).get('status','NO_STATUS')+' '+str(record.get('counts',{})),flush=True)
        return record
    records=[];write_summary(folder,records,manifest,label,stage)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending={executor.submit(one,m):m for m in selected}
        for future in as_completed(pending):
            records.append(future.result());write_summary(folder,records,manifest,label,stage)
    print('BATCH COMPLETE '+str(folder),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='action',required=True)
    prep=subs.add_parser('prepare');prep.add_argument('--archive',required=True,type=Path);prep.add_argument('--suite',required=True,type=Path)
    run=subs.add_parser('run');run.add_argument('--suite',required=True,type=Path);run.add_argument('--label',required=True);run.add_argument('--jobs',type=int,choices=[1,2],default=2)
    run.add_argument('--stage',choices=['analyze','all'],default='all');run.add_argument('--indices',type=int,nargs='+');run.add_argument('--target-length-mm',type=float)
    run.add_argument('--calibration-plan',type=Path)
    args=parser.parse_args();suite=args.suite.resolve()
    if args.action=='prepare':prepare(args.archive,suite)
    else:
        if not re.fullmatch('[a-zA-Z0-9_-]+',args.label):parser.error('Label must be a simple directory name.')
        run_batch(suite,args.label,args.jobs,args.stage,args.indices,args.target_length_mm,args.calibration_plan)


if __name__=='__main__':main()
