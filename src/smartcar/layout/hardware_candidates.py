from __future__ import annotations
import numpy as np
from smartcar.domain.assembly import placed
from smartcar.geometry.collision import aabb_distance,expand,overlaps
from smartcar.layout.scoring import hardware_score,total_score
from smartcar.layout.reservations import structural_bounds,footprint_conflict


def orientations(hw):
    if hw.role=="battery":
        dims=np.ptp(hw.bounding_box,axis=0)
        short,mid,long=np.argsort(dims)
        base=np.eye(3)[[mid,long,short]]
        if np.linalg.det(base)<0:base[0]*=-1
        out=[("flat_longitudinal",base),("flat_transverse",np.array([[0,-1,0],[1,0,0],[0,0,1]])@base),
             ("side_longitudinal",np.array([[0,0,1],[0,1,0],[-1,0,0]])@base)]
    else:
        # Derive PCB normal from the largest face, aligning it to +/-Z.
        normal=np.abs(np.array(hw.features["board_plane"]["normal"]))
        a=int(np.argmax(normal));remaining=[i for i in range(3) if i!=a]
        b=np.eye(3)[[remaining[0],remaining[1],a]]
        if np.linalg.det(b)<0:b[1]*=-1
        out=[]
        for sign in [1,-1]:
            if hw.role=="power_switch" and sign==1:continue
            support=hw.features.get("mount_support_sides",{}).get(str(sign),{})
            if not support.get("valid",False):continue
            flipped=np.diag([1,sign,sign])@b
            for n,rr in [("0",np.eye(3)),("90",np.array([[0,-1,0],[1,0,0],[0,0,1]])),("180",np.diag([-1,-1,1])),("270",np.array([[0,1,0],[-1,0,0],[0,0,1]]))]:
                out.append((f"horizontal_side{sign}_{n}",rr@flipped))
    return out


def generate_candidates(hw,volume,vehicle,obstacles,floor,profile,topology):
    output=[];counts={}
    for name,r in orientations(hw):
        ext=np.abs(r)@np.ptp(hw.bounding_box,axis=0)
        step=max(1,round(profile.candidate_pitch/volume.pitch))
        positions,clearances=volume.feasible_centers(ext,profile.rigid_clearance,bottom=floor+profile.rigid_clearance,stride=(step,step,1))
        counts[name]=dict(volume_feasible=len(positions),obstacle_feasible=0)
        if len(positions):
            # Wheel conflicts can depend on Z. Filter these before removing
            # vertically dominated positions in the common-tray architecture.
            low=positions-ext/2-profile.moving_clearance;high=positions+ext/2+profile.moving_clearance
            eligible=np.ones(len(positions),bool)
            for obstacle in obstacles:
                if 'wheel' in obstacle.definition.role:
                    reserved_low=low.copy()
                    if hw.role=='battery':
                        # An elevated tray requires material between the pack
                        # and the actual wheel-arch cut, not merely air between
                        # battery and tire bounding boxes.
                        from smartcar.geometry.wheel import arch_radius
                        rad=obstacle.definition.raw['rigid_body']['nominal_diameter_mm']/2
                        reserved_low[:,2]-=profile.nominal_wall+arch_radius(rad,profile)-rad-profile.moving_clearance
                    eligible&=~np.all((reserved_low<obstacle.bounds[1])&(high>obstacle.bounds[0]),axis=1)
            if hw.role=='power_switch':eligible&=positions[:,2]-ext[2]/2<=floor+profile.candidate_pitch+profile.rigid_clearance
            positions=positions[eligible];clearances=clearances[eligible]
            # For identical XY/orientation, all remaining support exclusions are
            # independent of height and the height score strictly increases.
            # The previous final diversity filter already discarded these poses.
            if len(positions):
                _,indices=np.unique(positions[:,:2],axis=0,return_index=True)
                counts[name]['dominated_vertical_samples_removed']=len(positions)-len(indices)
                positions=positions[indices];clearances=clearances[indices]
        for pos,clearance in zip(positions,clearances):
            inst=placed(hw,r,pos,hw.role,name)
            # Supports and removal paths also consume horizontal space. This
            # first topology is all on one serviceable tray; stacking is tried
            # separately using a removable bridge when implemented.
            margin=profile.support_wall+profile.rigid_clearance
            if any(footprint_conflict(inst,o,profile) for o in obstacles if o.definition.role=="drive_unit"):continue
            if any(overlaps(expand(inst.bounds,profile.moving_clearance),o.bounds) for o in obstacles if "wheel" in o.definition.role):continue
            if hw.role in ['main_controller','power_switch']:
                blocked=False
                for hole in hw.features.get('mounting_holes',[]):
                    location=r@np.asarray(hole['point'])+inst.transform[:3,3]
                    extent=profile.standoff_radius+profile.moving_clearance
                    support=np.array([[location[0]-extent,location[1]-extent,floor],[location[0]+extent,location[1]+extent,inst.bounds[1,2]]])
                    if any(overlaps(support,o.bounds) for o in obstacles if 'wheel' in o.definition.role):blocked=True;break
                if blocked:continue
            # Direct downward actuator access rejects blocked hardware poses.
            if hw.role=="power_switch":
                if abs(r[2,1]+1)>.001:continue
                if inst.bounds[0,2]>floor+profile.candidate_pitch+profile.rigid_clearance:continue
            score=hardware_score(inst.bounds,vehicle.bounds,hw.role)
            if topology=="battery_forward":
                score["region_preference"]=float(max(0,-pos[1])/vehicle.extents[1]) if hw.role=="battery" else float(max(0,pos[1])/vehicle.extents[1])
            elif topology=="battery_rear":
                score["region_preference"]=float(max(0,pos[1])/vehicle.extents[1]) if hw.role=="battery" else float(max(0,-pos[1])/vehicle.extents[1])
            inst.score=score;inst.minimum_clearance_mm=float(clearance-profile.nominal_wall)
            inst.constraints=dict(inside_designable_volume=True,conservative_box_erosion=True)
            output.append(inst);counts[name]["obstacle_feasible"]+=1
    output.sort(key=lambda c:total_score(c.score))
    # Retain orientation and spatial diversity for discrete search, not only the
    # best positions in one cramped region.
    unique=[]
    for c in output:
        center=c.bounds.mean(0)
        if all(c.selected_orientation!=p.selected_orientation or np.linalg.norm(center[:2]-p.bounds.mean(0)[:2])>=profile.candidate_pitch*1.4 for p in unique):unique.append(c)
        if len(unique)>=profile.candidate_limit:break
    return unique[:profile.candidate_limit],counts
