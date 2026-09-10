from __future__ import annotations
import argparse
import datetime
import shutil
import sys
import traceback
import os
from pathlib import Path
from smartcar.io import write_json,sha256
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.domain.hardware import load_kit
from smartcar.geometry.mesh import load_mesh,mesh_stats
from smartcar.geometry.repair import repair_volume
from smartcar.geometry.sdf import DesignableVolume
from smartcar.understanding.coordinate_frame import establish_frame
from smartcar.understanding.semantic_regions import separate_decorative_wheels
from smartcar.render import render_meshes

STAGES=["01_input","02_repaired","03_coordinate_frame","04_designable_volume","05_wheel_candidates","06_wheel_solution","07_hardware_candidates","08_layout","09_structure","10_openings","11_assembly","12_validation","output"]


_active_run=None


def project_root():
    # A batch may pin the implementation to a source snapshot while retaining
    # the project's input, cache and immutable run directories.
    return Path(os.environ.get('SMARTCAR_PROJECT_ROOT',Path(__file__).resolve().parents[2])).resolve()


class RunLog:
    def __init__(self,stream,file):self.stream=stream;self.file=file
    def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush()
    def flush(self):self.stream.flush();self.file.flush()


def execute():
    global _active_run
    root=project_root()
    p=argparse.ArgumentParser()
    p.add_argument("--appearance",type=Path,default=root/"source_bundle/smartcar-agent-foundation-v1/data/appearance/vehicle_appearance.stl")
    p.add_argument("--kit",type=Path,default=root/"source_bundle/smartcar-agent-foundation-v1/data/hardware/kit_manifest.json")
    p.add_argument("--profile",type=Path,default=root/"config/manufacturing.json")
    p.add_argument('--vehicle-policy',type=Path,default=root/'config/vehicle_design.json')
    p.add_argument("--stage",choices=["analyze","all"],default="all")
    p.add_argument("--run-id",default=None)
    p.add_argument('--max-scale',type=float,help='Maximum appearance scale searched after input calibration (overrides profile).')
    calibration=p.add_mutually_exclusive_group()
    calibration.add_argument('--input-unit',choices=['mm','cm','m','in'],default=None,help='Declared source coordinate unit (default assumes mm; STL does not store units).')
    calibration.add_argument('--target-length-mm',type=float,help='Set starting vehicle-frame Y length in mm before the independent layout scale search.')
    args=p.parse_args()
    run=root/"runs"/(args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if run.exists(): raise ValueError("RUN_ALREADY_EXISTS: use a new run id")
    for stage in STAGES:(run/stage).mkdir(parents=True)
    _active_run=run
    logfile=(run/'execution.log').open('w',encoding='utf-8',buffering=1)
    sys.stdout=RunLog(sys.stdout,logfile)
    sys.stderr=RunLog(sys.stderr,logfile)
    write_json(run/'run_status.json',dict(status='RUNNING',pid=os.getpid(),release_ready=False))
    snapshot=run/"01_input/source_snapshot"
    shutil.copytree(Path(__file__).resolve().parents[1],snapshot,ignore=shutil.ignore_patterns("__pycache__","*.egg-info"))
    # All lazy imports during this run resolve to this immutable implementation
    # snapshot, so editing the project cannot mix source versions in one run.
    for name,module in list(sys.modules.items()):
        if (name=="smartcar" or name.startswith("smartcar.")) and hasattr(module,"__path__"):
            package_path=snapshot/Path(*name.split("."))
            if package_path.is_dir():module.__path__=[str(package_path)]
    profile=ManufacturingProfile.load(args.profile)
    from smartcar.domain.vehicle_policy import VehiclePolicy
    policy=VehiclePolicy.load(args.vehicle_policy)
    write_json(run/'01_input/vehicle_policy.json',policy.to_dict())
    if args.max_scale is not None:
        import math
        from smartcar.geometry.errors import GeometryInputError
        if not math.isfinite(args.max_scale) or args.max_scale<1:
            raise GeometryInputError('INVALID_MAX_SCALE','--max-scale must be finite and at least 1.')
        profile=ManufacturingProfile(profile.values|dict(maximum_scale=args.max_scale))
    shutil.copy2(args.appearance,run/"01_input/appearance.stl")
    provenance=dict(appearance=str(args.appearance.resolve()),sha256=sha256(args.appearance),kit=str(args.kit.resolve()),manufacturing=profile.values)
    write_json(run/"01_input/provenance.json",provenance)
    print(f"RUN {run}",flush=True)
    source=load_mesh(args.appearance)
    write_json(run/'01_input/source_geometry.json',dict(coordinate_unit='unspecified by STL',bounds=source.bounds,dimensions=source.extents,triangles=len(source.faces)))
    from smartcar.understanding.input_scale import prepare_appearance
    raw,normalized,frame,preparation=prepare_appearance(source,profile,args.input_unit,args.target_length_mm,policy)
    write_json(run/'01_input/input_preparation.json',preparation)
    provenance['input_preparation']=preparation;write_json(run/'01_input/provenance.json',provenance)
    print(f"INPUT scale_to_mm={preparation['source_coordinate_to_mm_factor']:.9g} basis={preparation['basis']}",flush=True)
    from smartcar.geometry.repair import check_voxel_budget
    check_voxel_budget(normalized.extents,profile.voxel_pitch,getattr(profile,'maximum_voxel_cells',80000000))
    write_json(run/"01_input/mesh_analysis.json",mesh_stats(raw))
    render_meshes(run/"01_input/original.png",[("body",raw)],"Original appearance (source axes)")
    normalized.export(run/"03_coordinate_frame/normalized.stl")
    write_json(run/"03_coordinate_frame/frame.json",frame)
    render_meshes(run/"03_coordinate_frame/frame.png",[("body",normalized)],"Automatic vehicle frame")
    print(f"FRAME {normalized.extents}",flush=True)
    semantic_body,semantic=separate_decorative_wheels(normalized,profile.voxel_pitch)
    from smartcar.understanding.axle_anchors import detect_axle_anchors
    evidence=detect_axle_anchors(normalized,semantic,profile.voxel_pitch,policy)
    write_json(run/'03_coordinate_frame/axle_evidence.json',evidence)
    print('SOURCE AXLES '+str([a['y_mm'] for a in evidence['axles']])+' '+evidence['confidence'],flush=True)
    write_json(run/"03_coordinate_frame/semantic_regions.json",semantic)
    semantic_body.export(run/"03_coordinate_frame/semantic_body.stl")
    repaired,occ,origin,repair=repair_volume(semantic_body,profile.voxel_pitch,profile.minimum_wall,getattr(profile,'maximum_voxel_cells',80000000))
    repaired.export(run/'02_repaired/before_wheel_regions.stl')
    from smartcar.geometry.wheelwell_envelope import reconstruct_wheel_regions
    from smartcar.geometry.repair import occupancy_mesh
    redesigned,region_record=reconstruct_wheel_regions(occ,origin,profile.voxel_pitch,evidence,policy)
    if region_record.get('status')=='FAIL':
        # Strategy feedback changes the geometry, never the exterior-change
        # allowance. Crown-only support is a narrower alternative when extending
        # the neighbouring panels would replace too much of the appearance.
        previous=region_record
        redesigned,region_record=reconstruct_wheel_regions(occ,origin,profile.voxel_pitch,evidence,policy,width_strategy='crown')
        region_record['alternatives']=[previous]
        region_record['strategy_reason']='flank extension exceeded the unchanged exterior addition budget; test measured crown-only envelope'
    if region_record.get('applied'):
        if (redesigned&~occ).any():occupancy_mesh(redesigned&~occ,origin,profile.voxel_pitch).export(run/'02_repaired/wheel_region_added.stl')
        if (occ&~redesigned).any():occupancy_mesh(occ&~redesigned,origin,profile.voxel_pitch).export(run/'02_repaired/wheel_region_removed.stl')
        region_record['accepted']=region_record['status']!='FAIL'
        if region_record['accepted']:
            occ=redesigned;repaired=occupancy_mesh(occ,origin,profile.voxel_pitch)
        else:
            region_record['warning']+=' Proposed addition exceeds policy; original repaired envelope retained.'
    write_json(run/'02_repaired/wheel_region_reconstruction.json',region_record)
    repair['wheel_region_reconstruction']=region_record
    repaired.export(run/"02_repaired/envelope.stl")
    from smartcar.validation.appearance import measured_surface_deviation
    repair["source_to_repaired_deviation"]=measured_surface_deviation(semantic_body,repaired,profile)
    write_json(run/"02_repaired/repair.json",repair|dict(mesh=mesh_stats(repaired)))
    render_meshes(run/"02_repaired/envelope.png",[("body",repaired)],"Reconstructed exterior envelope")
    volume=DesignableVolume(occ,origin,profile.voxel_pitch,profile.nominal_wall)
    if volume.allowed.any():
        vm=volume.mesh();vm.export(run/"04_designable_volume/designable.stl")
        render_meshes(run/"04_designable_volume/designable.png",[("volume",vm)],"Potential internal design volume")
    else:
        # An empty initial erosion can still become usable during the explicitly
        # configured layout scale search. Do not manufacture a fake surface or
        # send an empty field to marching cubes.
        print('NO_DESIGNABLE_VOLUME_AT_INPUT_SCALE: retaining field for bounded layout scale search',flush=True)
    write_json(run/"04_designable_volume/volume.json",dict(pitch=volume.pitch,uncertainty=volume.error,allowed_mm3=float(volume.allowed.sum()*volume.pitch**3),wall=volume.wall))
    print("Importing authoritative hardware CAD",flush=True)
    kit,manifest=load_kit(args.kit,root/"cache/hardware",profile)
    write_json(run/"01_input/hardware_definitions.json",{k:v.summary() for k,v in kit.items()})
    for role,hw in kit.items():
        print(f"HARDWARE {role}: {hw.bounding_box.tolist()} solids={hw.features.get('solid_count')}",flush=True)
        hw.mesh.export(run/f"01_input/{role}.stl")
    if args.stage == "analyze":
        write_json(run/'run_status.json',dict(status='ANALYSIS_COMPLETE',release_ready=False))
        print("ANALYSIS_COMPLETE",flush=True);return
    from smartcar.agent.planner import design
    design(run,normalized,repaired,volume,kit,manifest,profile,evidence,policy)


def worker_main():
    try:execute()
    except Exception as error:
        from smartcar.geometry.errors import GeometryInputError
        if _active_run is not None:
            details=dict(error_code=error.code,measurements=error.details) if isinstance(error,GeometryInputError) else {}
            write_json(_active_run/'run_status.json',dict(status='ERROR',error_type=type(error).__name__,error=str(error),release_ready=False,**details))
            (_active_run/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if isinstance(error,GeometryInputError):
            print(str(error),file=sys.stderr,flush=True)
            raise SystemExit(2)
        raise


def main():
    if os.environ.get('SMARTCAR_PIPELINE_WORKER')=='1' or '--help' in sys.argv or '-h' in sys.argv:
        return worker_main()
    import subprocess
    from smartcar.agent.runtime import record_worker_exit
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--run-id')
    known,_=parser.parse_known_args()
    run_id=known.run_id or datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if Path(run_id).name!=run_id or run_id in ['.','..']:raise ValueError('RUN_ID_MUST_BE_ONE_DIRECTORY_NAME')
    root=project_root();run=root/'runs'/run_id
    # Refuse an existing run in the parent too: it must never rewrite an older
    # completed status merely because the worker rejects an existing run ID.
    if run.exists():raise ValueError('RUN_ALREADY_EXISTS: use a new run id')
    arguments=sys.argv[1:]+([] if known.run_id else ['--run-id',run_id])
    env=os.environ.copy();env['SMARTCAR_PIPELINE_WORKER']='1'
    completed=subprocess.run([sys.executable,'-m','smartcar.pipeline',*arguments],env=env)
    record_worker_exit(run,completed.returncode)
    if completed.returncode:raise SystemExit(completed.returncode)


if __name__ == "__main__": main()
