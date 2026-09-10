from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates
from smartcar.domain.assembly import HardwareInstance,placed
from smartcar.geometry.collision import aabb_distance


def rotation_for_shaft(hw,side,long_sign,long_axis=1):
    # Construct an orthonormal frame from the CAD shaft and rigid principal extent.
    axis=np.asarray(hw.features["shaft"]["axis"])
    source_long=np.eye(3)[np.argmax(np.ptp(hw.bounding_box,axis=0))]
    source_up=np.cross(axis,source_long)
    if long_axis not in [1,2]:raise ValueError('Motor long axis must be perpendicular to the vehicle axle.')
    target_axis=np.array([side,0,0]);target_long=np.zeros(3);target_long[long_axis]=long_sign
    target_up=np.cross(target_axis,target_long)
    return np.stack([target_axis,target_long,target_up],axis=1)@np.stack([axis,source_long,source_up])


def drive_instances(hw,wheel,center,long_sign,side,long_axis=1):
    r=rotation_for_shaft(hw,side,long_sign,long_axis)
    shaft=hw.features["shaft"]
    width=wheel.raw["rigid_body"]["nominal_width_mm"]
    axis=np.array([side,0,0])
    # Fixed module mating: socket starts on the inner wheel plane; this is an
    # explicit axial assumption until authoritative hub CAD is supplied.
    tip=np.asarray(center)-axis*(width/2-shaft["engagement"])
    t=tip-r@np.asarray(shaft["tip"])
    matrix=np.eye(4);matrix[:3,:3]=r;matrix[:3,3]=t
    corners=np.array(np.meshgrid(*hw.bounding_box.T,indexing="ij")).reshape(3,-1).T@r.T+t
    suffix="left" if side>0 else "right"
    motor=HardwareInstance("drive_"+suffix,hw,matrix,f"fixed_drive_long_{'y' if long_axis==1 else 'z'}_{long_sign:+d}",np.array([corners.min(0),corners.max(0)]),
                           metadata=dict(module="drive_module_"+suffix,motor_long_axis=long_axis,motor_long_sign=long_sign,shaft_axis=axis,shaft_tip=tip,wheel_center=center,wheel_width_mm=width,engagement_mm=shaft["engagement"],hub_axial_assumption="socket entrance at inner nominal wheel face"))
    wh=placed(wheel,np.eye(3),center,"wheel_drive_"+suffix,"shaft_fixed",module="drive_module_"+suffix)
    return motor,wh


def wheel_candidates(volume,vehicle,kit,profile,floor,diagnostics=None,candidate_filter=None):
    hw=kit["drive_unit"];wheel=kit["drive_wheel"]
    r=wheel.raw["rigid_body"]["nominal_diameter_mm"]/2
    width=wheel.raw["rigid_body"]["nominal_width_mm"]
    bounds=vehicle.bounds; rows=[];valid=[]
    if diagnostics is not None:
        diagnostics.update(vehicle_dimensions_mm=vehicle.extents,floor_mm=floor,
                           wheel_diameter_mm=2*r,orientations=[],
                           interpretation='Sequential hard filters; counts are rejected candidates, never score penalties.')
    # Hundreds to thousands of complete bilateral drive-module candidates, each
    # based on eroded current geometry, never a retained wheel center.
    for long_axis,long_sign in [(1,-1),(1,1),(2,-1),(2,1)]:
        rot=rotation_for_shaft(hw,1,long_sign,long_axis)
        ext=np.abs(rot)@np.ptp(hw.bounding_box,axis=0)
        step=max(1,round(profile.candidate_pitch/volume.pitch))
        stats=dict(long_axis=long_axis,long_sign=long_sign,motor_extents_mm=ext,lateral_range_rejected=0,wheel_face_opening_rejected=0,end_wall_rejected=0,
                   ground_rejected=0,motor_pair_collision_rejected=0,mirror_containment_rejected=0,accepted=0,
                   best_ground_clearance_mm=None)
        centers,dist=volume.feasible_centers(ext,profile.rigid_clearance,bottom=floor+profile.rigid_clearance,stride=(step,step,1),diagnostics=stats)
        if diagnostics is not None:diagnostics['orientations'].append(stats)
        for center,clearance in zip(centers,dist):
            temp=placed(hw,rot,center,"tmp","drive")
            tip=rot@np.asarray(hw.features["shaft"]["tip"])+temp.transform[:3,3]
            wc=tip+np.array([width/2-hw.features["shaft"]["engagement"],0,0])
            if wc[0]<=0 or wc[0]>bounds[1,0]+width:
                stats['lateral_range_rejected']+=1;continue
            from smartcar.geometry.wheel import wheel_end_margin
            end_margin=wheel_end_margin(r,profile)
            if wc[1]-r-end_margin<bounds[0,1] or wc[1]+r+end_margin>bounds[1,1]:
                stats['end_wall_rejected']+=1;continue
            ground=wc[2]-r
            ground_clearance=float(floor-profile.bottom_thickness-ground)
            if stats['best_ground_clearance_mm'] is None or ground_clearance>stats['best_ground_clearance_mm']:
                stats['best_ground_clearance_mm']=ground_clearance
                stats['best_ground_candidate']=dict(wheel_center=wc,motor_bounds=temp.bounds,required_mm=profile.rigid_clearance)
            if ground_clearance<profile.rigid_clearance:
                stats['ground_rejected']+=1;continue
            instances=[]
            for side in [1,-1]:instances.extend(drive_instances(hw,wheel,wc*np.array([side,1,1]),long_sign,side,long_axis))
            motors=[i for i in instances if i.definition.role=="drive_unit"]
            if aabb_distance(motors[0].bounds,motors[1].bounds)<profile.rigid_clearance:
                stats['motor_pair_collision_rejected']+=1;continue
            # Mirrored partner also needs a full-volume constraint, since input
            # appearance is not assumed exactly symmetric.
            c2=volume.box_clearance(motors[1].bounds,step=volume.pitch*3)
            if c2<profile.nominal_wall+profile.rigid_clearance:
                stats['mirror_containment_rejected']+=1;continue
            from smartcar.geometry.wheel import wheel_side_exposure
            exposure=[wheel_side_exposure(volume,wc*np.array([side,1,1]),r,width,side,maximum_recess=profile.nominal_wall) for side in [1,-1]]
            exposed_fraction=min(e['exposed_fraction'] for e in exposure)
            stats['best_side_opening_fraction']=max(stats.get('best_side_opening_fraction',0.),exposed_fraction)
            if exposed_fraction<getattr(profile,'minimum_wheel_exposed_fraction',.5):
                stats['wheel_face_opening_rejected']+=1;continue
            # Estimate area damage by samples of the rotating cylinder inside
            # appearance; actual sweep is cut and independently checked later.
            wmesh=instances[1].mesh()
            inside=volume.query(wmesh.vertices)>0
            penetration=float(inside.mean())
            score=dict(appearance_damage=penetration,geometric_height=float(wc[2]/vehicle.extents[2]),
                       track_exposure=float((wc[0]-bounds[1,0])/width),wheelbase_reward=float(abs(wc[1])/vehicle.extents[1]),
                       mount_height=float((motors[0].bounds[1,2]-floor)/vehicle.extents[2]),side_exposure_fraction=exposed_fraction)
            total=5*penetration+score["geometric_height"]+.4*score["track_exposure"]-2*score["wheelbase_reward"]+.25*score['mount_height']
            entry=dict(index=len(rows),center=wc,ground_z=ground,long_axis=long_axis,long_sign=long_sign,drive_region="front" if wc[1]>0 else "rear",minimum_exterior_distance=float(min(clearance,c2)),wheel_side_exposure=exposure,score=score,total=total)
            if candidate_filter is not None and not candidate_filter(entry,instances):
                stats['additional_hard_constraints_rejected']=stats.get('additional_hard_constraints_rejected',0)+1
                continue
            for inst in instances:inst.score=dict(score)
            rows.append(entry);valid.append((entry,instances))
            stats['accepted']+=1
    valid.sort(key=lambda x:x[0]["total"])
    # Position diversity avoids exhausting retries on nearly identical centers.
    selected=[]
    for x in valid:
        if all(x[0]['long_axis']!=y[0]['long_axis'] or x[0]['long_sign']!=y[0]['long_sign'] or np.linalg.norm(x[0]["center"]-y[0]["center"])>=profile.candidate_pitch*1.5 for y in selected):selected.append(x)
    if diagnostics is not None:diagnostics.update(accepted=len(rows),diverse=len(selected))
    return selected,rows


def passive_candidates(volume,vehicle,kit,profile,drive_solution):
    entry,instances=drive_solution; rear=entry["center"]
    w=kit["passive_wheel"];r=w.raw["rigid_body"]["nominal_diameter_mm"]/2
    choices=[]
    from smartcar.geometry.wheel import wheel_end_margin
    margin=wheel_end_margin(r,profile)
    for y in np.arange(vehicle.bounds[0,1]+r+margin,vehicle.bounds[1,1]-r-margin,profile.candidate_pitch):
        if y*rear[1]>0 or abs(y-rear[1])<2*r+profile.moving_clearance:continue
        pair=[placed(w,np.eye(3),[s*rear[0],y,rear[2]],"wheel_passive_"+("left" if s>0 else "right"),"parallel_axle") for s in [1,-1]]
        damage=sum(float((volume.query(p.mesh().vertices)>0).mean()) for p in pair)
        for inst in pair:inst.score=dict(appearance_damage=damage/2,wheelbase_reward=float(abs(y-rear[1])/vehicle.extents[1]))
        choices.append((damage-2*abs(y-rear[1])/vehicle.extents[1],pair))
    return min(choices,key=lambda x:x[0])[1] if choices else []
