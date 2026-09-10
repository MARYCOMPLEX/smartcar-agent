from __future__ import annotations
import time
import shutil
import numpy as np
from smartcar.io import write_json,write_csv,read_json
from smartcar.geometry.sdf import DesignableVolume,resample_designable_volume
from smartcar.layout.axle_solver import joint_axle_candidates
from smartcar.layout.hardware_candidates import generate_candidates
from smartcar.layout.solver import solve_layout
from smartcar.render import candidate_plot
from smartcar.understanding.design_frame import adopt_ground


def select_floor(volume,profile):
    mask=volume.occupancy;nx,ny,nz=mask.shape
    # Robust belly surface from the central footprint excludes decorative wheels
    # at lateral edges. No vehicle-specific cut height is used.
    region=mask[int(.3*nx):int(.7*nx),int(.1*ny):int(.9*ny)]
    first=region.argmax(2)[region.any(2)]
    belly=volume.origin[2]+float(np.quantile(first,.6))*volume.pitch
    return belly+profile.bottom_thickness


def design(run,normalized,repaired,base_volume,kit,manifest,profile,evidence,policy):
    from smartcar.structure.synthesis import synthesize
    from smartcar.assembly.planner import plan_assembly
    from smartcar.validation.validator import validate
    from smartcar.output import export_output
    history=[];best=None;best_key=None;evaluations=0;seen=set();stop=False;resource_failures=[]
    frame=read_json(run/"03_coordinate_frame/frame.json")
    # Policy contains topology/orientation/region choices only. It never supplies
    # millimeter placements or edits solver results. External agents can consume
    # failure codes and extend this strategy queue without owning geometry truth.
    scales=np.round(np.arange(1,profile.maximum_scale+.001,.1),4)
    scale_search=dict(maximum_scale=profile.maximum_scale,step=.1,attempted_scales=[],
                      first_drive_feasible_scale=None,first_layout_feasible_scale=None,
                      selected_scale=None,global_minimum_proven=False,
                      method='bounded discrete scale, orientation, anchor and topology search; sampling failure is not an impossibility proof')
    for scale in scales:
        scale_search['attempted_scales'].append(float(scale))
        write_json(run/'scale_search.json',scale_search)
        vehicle=repaired.copy().apply_scale(scale)
        if scale==1:
            volume=base_volume
            volume_record=dict(method='initial repaired volume',pitch_mm=volume.pitch,distance_uncertainty_mm=volume.error)
        else:
            volume,volume_record=resample_designable_volume(vehicle,profile.voxel_pitch,profile.nominal_wall,getattr(profile,'maximum_voxel_cells',80000000))
        write_json(run/f'04_designable_volume/scale-{scale:.3f}.json',volume_record)
        floor=select_floor(volume,profile)
        print(f"SEARCH scale={scale:.3f} floor={floor:.3f}",flush=True)
        wheel_diagnostics={}
        wheels,rows=joint_axle_candidates(volume,vehicle,kit,profile,floor,evidence,scale,policy,wheel_diagnostics)
        write_json(run/f"05_wheel_candidates/diagnostics-scale-{scale:.3f}.json",wheel_diagnostics)
        write_json(run/f"05_wheel_candidates/scale-{scale:.3f}.json",rows)
        write_csv(run/f"05_wheel_candidates/scale-{scale:.3f}.csv",rows)
        print(f"WHEEL candidates={len(rows)} diverse={len(wheels)}",flush=True)
        if not wheels:
            history.append(dict(scale=scale,failure="JOINT_AXLE_NO_FEASIBLE_POSE",wheel_diagnostics=wheel_diagnostics,action="test next scale after sampled orientations and measured axle regions exhausted"));write_json(run/"agent_trace.json",history);continue
        if scale_search['first_drive_feasible_scale'] is None:scale_search['first_drive_feasible_scale']=float(scale)
        candidate_plot(run/"05_wheel_candidates/candidates.png",vehicle,[("drive modules",[r["center"] for r in rows])],f"Wheel candidates at scale {scale:.2f}")
        cached_groups={}
        for attempt in range(min(profile.layout_attempts_per_scale,len(wheels)*3)):
            choice=wheels[attempt//3];topology=["battery_forward","battery_rear","spatial_free"][attempt%3]
            wheel_record,anchors=choice
            anchor_id=attempt//3
            if anchor_id not in cached_groups:
                groups={};counts={}
                for role in ["battery","main_controller","power_switch"]:
                    groups[role],counts[role]=generate_candidates(kit[role],volume,vehicle,anchors,floor,profile,"spatial_free")
                cached_groups[anchor_id]=(groups,counts)
            groups,counts=cached_groups[anchor_id]
            for role,cs in groups.items():
                for inst in cs:
                    y=inst.bounds.mean(0)[1]
                    sign=1 if (topology=="battery_forward")== (role=="battery") else -1
                    inst.score["region_preference"]=0. if topology=="spatial_free" else float(max(0,-sign*y)/vehicle.extents[1])
            print(f"LAYOUT {topology} candidates="+str({k:len(v) for k,v in groups.items()}),flush=True)
            layout,solver=solve_layout(groups,profile)
            event=dict(scale=scale,attempt=attempt,topology=topology,wheel_candidate=wheel_record,counts=counts,solver=solver)
            history.append(event);write_json(run/"agent_trace.json",history)
            write_json(run/f"07_hardware_candidates/scale-{scale:.3f}-attempt-{attempt}.json",{k:[i.to_dict() for i in v] for k,v in groups.items()})
            if layout is None:
                event["failure"]="INTERIOR_LAYOUT_INFEASIBLE";event["action"]="change region/topology, then drive anchor, then scale";continue
            if scale_search['first_layout_feasible_scale'] is None:scale_search['first_layout_feasible_scale']=float(scale)
            write_json(run/'scale_search.json',scale_search)
            signature=tuple(tuple(np.round(i.transform.ravel(),6)) for i in anchors+layout)
            if signature in seen:
                event["action"]="duplicate geometric layout skipped; try next strategy";continue
            seen.add(signature);evaluations+=1
            iteration=run/"design_iterations"/f"{evaluations:02d}-scale-{scale:.3f}"
            for stage in ["03_coordinate_frame","05_wheel_candidates","06_wheel_solution","07_hardware_candidates","08_layout","09_structure","10_openings","11_assembly","12_validation","output"]:(iteration/stage).mkdir(parents=True,exist_ok=True)
            event["design_iteration"]=str(iteration.relative_to(run))
            candidate_plot(iteration/"07_hardware_candidates/candidates.png",vehicle,[(k,[i.bounds.mean(0) for i in v]) for k,v in groups.items()],"Interior candidate centers")
            for role,cs in groups.items():candidate_plot(iteration/f"07_hardware_candidates/{role}_candidates.png",vehicle,[(role,[i.bounds.mean(0) for i in cs])],role+" feasible candidates")
            eval_vehicle,eval_volume,instances,eval_floor=adopt_ground(vehicle,volume,anchors+layout,floor,wheel_record["ground_z"])
            ground_shift=np.eye(4);ground_shift[2,3]=-wheel_record["ground_z"]
            source_scale=np.diag([scale,scale,scale,1.])
            write_json(iteration/"03_coordinate_frame/design_frame.json",dict(input_to_design=ground_shift@source_scale@np.array(frame["input_to_vehicle"]),ground_plane=[0,0,1,0],scale=scale))
            eval_wheel=dict(wheel_record,center=np.asarray(wheel_record["center"])+ground_shift[:3,3],ground_z=0.)
            if 'passive_center' in eval_wheel:
                eval_wheel['passive_center']=np.asarray(eval_wheel['passive_center'])+ground_shift[:3,3]
            eval_wheel['coordinate_space']='design frame with physical ground Z=0'
            if 'axle_targets' in eval_wheel:
                eval_wheel['axle_targets']=dict(eval_wheel['axle_targets'],bounds_mm=np.asarray(eval_wheel['axle_targets']['bounds_mm'])+ground_shift[:3,3])
            # Geometric support positions are solver results, not hand edits.
            write_json(iteration/"08_layout/layout.json",[i.to_dict() for i in instances])
            write_json(iteration/"06_wheel_solution/selected.json",eval_wheel)
            print(f"SYNTHESIS evaluation={evaluations}",flush=True)
            try:
                parts,records=synthesize(eval_vehicle,instances,eval_floor,profile,iteration,eval_volume,kit.get("fastener_reference"))
            except ValueError as error:
                event["failure"]=str(error);event["action"]="opening/geometry infeasible; return to pose search"
                from smartcar.geometry.errors import GeometryInputError
                if isinstance(error,GeometryInputError):
                    event['error_code']=error.code;event['measurements']=error.details
                    if error.code=='VOXEL_GRID_LIMIT':
                        resource_failures.append(dict(scale=float(scale),iteration=str(iteration.relative_to(run)),**error.details))
                        event['action']='stage resource budget exceeded; physical feasibility remains undetermined'
                write_json(run/"agent_trace.json",history)
                if evaluations>=profile.maximum_geometry_evaluations:stop=True;break
                continue
            print("ASSEMBLY validation",flush=True)
            plan=plan_assembly(instances,parts,eval_vehicle,profile,records)
            print("GEOMETRY validation",flush=True)
            report=validate(parts,instances,records,plan,eval_volume,profile,0.,frame,scale,full=True)
            from smartcar.validation.axle_layout import validate_axle_layout
            report['checks'].extend(validate_axle_layout(instances,evidence,scale,profile,policy,eval_volume))
            region_path=run/'02_repaired/wheel_region_reconstruction.json'
            if region_path.exists():
                region_record=read_json(region_path)
                report['checks'].append(dict(check='appearance_wheel_region_reconstruction',status='WARNING',measurements=region_record,
                    reason='measured local exterior redesign; source and changed volumes retained for appearance review'))
            report['counts']={s:sum(c['status']==s for c in report['checks']) for s in ['PASS','FAIL','WARNING']}
            report['blocking_checks']=[c['check'] for c in report['checks'] if c['status']=='FAIL']
            report['status']='FAIL' if report['counts']['FAIL'] else 'WARNING'
            from smartcar.validation.appearance import measured_surface_deviation
            from smartcar.geometry.solid import to_mesh
            deviation=measured_surface_deviation(eval_vehicle,to_mesh(parts["body"]),profile,eval_floor+profile.sliding_clearance)
            report["checks"].append(dict(check="appearance_preservation",status="WARNING",measurements=deviation,reason="measured against repaired scaled exterior; wheel/shaft/access apertures included; semantic preservation requires review"))
            report["counts"]["WARNING"]+=1
            from smartcar.layout.measured_result import attach_validation
            attach_validation(instances,report)
            write_json(iteration/"08_layout/layout.json",[i.to_dict() for i in instances])
            from smartcar.validation.exports import validate_stl_exports
            report=validate_stl_exports(report,iteration/"09_structure",parts,prefix="intermediate_STL:")
            write_json(iteration/"12_validation/validation_report.json",report)
            write_json(iteration/"11_assembly/assembly_plan.json",plan)
            event["validation_counts"]=report["counts"];event["blocking_checks"]=report["blocking_checks"]
            event["action"]="change region/topology/anchor and regenerate formal geometry" if report["counts"]["FAIL"] else "geometric candidate retained; unknown physical inputs remain explicit"
            write_json(run/"agent_trace.json",history)
            key=(report["counts"]["FAIL"],scale)
            if best_key is None or key<best_key:
                best_key=key;best=(parts,instances,plan,report,records,scale,iteration)
            print("VALIDATION "+str(report["counts"]),flush=True)
            if report["counts"]["FAIL"]==0 or evaluations>=profile.maximum_geometry_evaluations:
                stop=True;break
        if stop:break
    write_json(run/"agent_trace.json",history)
    if best is not None:
        parts,instances,plan,report,records,scale,iteration=best
        scale_search['selected_scale']=float(scale)
        scale_search['selected_intermediate_validation_counts']=dict(report['counts'])
        write_json(run/'scale_search.json',scale_search)
        for directory in iteration.iterdir():
            if directory.is_dir() and directory.name!="output":shutil.copytree(directory,run/directory.name,dirs_exist_ok=True)
        export_output(run,parts,instances,plan,report,records,manifest,profile,scale,0.,history,kit)
        scale_search['selected_validation_counts']=dict(report['counts'])
        write_json(run/'scale_search.json',scale_search)
        write_json(run/"run_status.json",dict(status=report["status"],output=str(run/"output"),release_ready=False,selected_iteration=str(iteration.relative_to(run)),geometry_evaluations=evaluations))
        print(f"OUTPUT {run/'output'}",flush=True)
        return
    write_json(run/'scale_search.json',scale_search)
    if resource_failures:
        write_json(run/"run_status.json",dict(status="RESOURCE_LIMIT",reason="sampled layouts reached synthesis but exceeded stage resource budgets",release_ready=False,scale_search=scale_search,resource_failures=resource_failures))
        print('RESOURCE_LIMIT: no completed design; inspect run_status.json for stage grid requirements. Physical infeasibility is not established.',flush=True)
        return
    write_json(run/"run_status.json",dict(status="INFEASIBLE",reason="no feasible sampled layout up to configured scale",release_ready=False,scale_search=scale_search))
    print(f"INFEASIBLE within configured scale limit {profile.maximum_scale:g}; inspect scale_search.json and wheel diagnostics. This is not a proof of geometric impossibility.",flush=True)
