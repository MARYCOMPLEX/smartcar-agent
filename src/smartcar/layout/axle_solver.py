"""Joint search for both axles before the interior component layout.

All poses originate in sampled geometry. Source axle regions are constraints,
not hand-authored wheel centers. Fixed motor/wheel transforms are supplied by
the drive-module generator; passive track may follow a tapered body separately.
"""
from __future__ import annotations
import numpy as np
from smartcar.domain.assembly import placed
from smartcar.domain.vehicle_policy import VehiclePolicy
from smartcar.geometry.wheel import wheel_side_exposure, wheel_face_points, wheel_end_margin
from smartcar.layout.wheel_candidates import wheel_candidates
from smartcar.understanding.axle_anchors import scaled_axle_targets
from smartcar.geometry.vehicle_section import axle_body_halfwidth


def _local_halfwidth(volume, center, radius, body_bounds):
    return axle_body_halfwidth(volume,center,radius,body_bounds)


def _damage(volume, center, radius, width):
    points=wheel_face_points(center,radius,width,1,samples=96)
    fractions=[]
    for x in np.linspace(center[0]-width/2,center[0]+width/2,3):
        points[:,0]=x;fractions.append(float((volume.query(points)>0).mean()))
    return float(np.mean(fractions))


def joint_axle_candidates(volume,vehicle,kit,profile,floor,evidence,scale,policy=None,diagnostics=None):
    policy=policy or VehiclePolicy();diagnostics={} if diagnostics is None else diagnostics
    radius=kit['drive_wheel'].raw['rigid_body']['nominal_diameter_mm']/2
    passive=kit['passive_wheel'];pr=passive.raw['rigid_body']['nominal_diameter_mm']/2
    width=passive.raw['rigid_body']['nominal_width_mm']
    target=scaled_axle_targets(evidence,scale,vehicle.bounds,radius,profile,policy)
    diagnostics['axle_targets']=target
    diagnostics['joint_rejections']={k:0 for k in ['source_axle_band','fastener_ground','drive_retainer_collision','drive_track','no_passive_pair']}
    diagnostics['passive_rejections']={k:0 for k in ['wheelbase','source_axle_band','side_exposure','overhang_balance']}
    accepted=[];rows=[];bounds=vehicle.bounds;length=vehicle.extents[1]
    local_cache={}
    def local_width(wc,r):
        key=(float(wc[1]),float(wc[2]),r)
        if key not in local_cache:local_cache[key]=_local_halfwidth(volume,wc,r,bounds)
        return local_cache[key]
    def drive_filter(entry,drive):
        wc=np.asarray(entry['center']);error=float(np.min(abs(np.asarray(target['target_y_mm'])-wc[1])))
        reason=None
        if target['measured'] and error>target['tolerance_mm']:reason='source_axle_band'
        elif floor-profile.bottom_thickness-entry['ground_z']<profile.screw_fit['head_height']+profile.rigid_clearance:reason='fastener_ground'
        if reason is None:
            from smartcar.layout.reservations import structural_footprints
            motors=[i for i in drive if i.definition.role=='drive_unit']
            a,b=[structural_footprints(i,profile)[1] for i in motors]
            if np.all((a[0]<b[1]+profile.rigid_clearance)&(b[0]<a[1]+profile.rigid_clearance)):reason='drive_retainer_collision'
        if reason is None and wc[0]+width/2<local_width(wc,radius)-profile.nominal_wall:reason='drive_track'
        if reason is not None:
            diagnostics['joint_rejections'][reason]+=1;return False
        # Source-correspondence ordering must precede the diversity reduction.
        # Otherwise a central inboard pose can suppress a nearby usable pose.
        entry['total']=float(4*error/target['tolerance_mm']+.35*abs(wc[0]-local_width(wc,radius))/width+.15*wc[2]/vehicle.extents[2])
        return True
    drives,drive_rows=wheel_candidates(volume,vehicle,kit,profile,floor,diagnostics,drive_filter)
    diagnostics['drive_only_candidates']=len(drive_rows)
    margin=wheel_end_margin(pr,profile)
    ys=np.arange(bounds[0,1]+pr+margin,bounds[1,1]-pr-margin,profile.candidate_pitch)
    # Include exact geometric feature projections in addition to the global
    # lattice. These are measured from this input, never stored design poses.
    ys=np.unique(np.concatenate([ys,np.asarray(target['target_y_mm'])]))
    ys=ys[(ys-pr-margin>=bounds[0,1])&(ys+pr+margin<=bounds[1,1])]
    for entry,drive in drives:
        wc=np.asarray(entry['center']);nearest=int(np.argmin(abs(np.asarray(target['target_y_mm'])-wc[1])))
        if target['measured'] and abs(wc[1]-target['target_y_mm'][nearest])>target['tolerance_mm']:
            diagnostics['joint_rejections']['source_axle_band']+=1;continue
        if floor-profile.bottom_thickness-entry['ground_z']<profile.screw_fit['head_height']+profile.rigid_clearance:
            diagnostics['joint_rejections']['fastener_ground']+=1;continue
        from smartcar.layout.reservations import structural_footprints
        motors=[i for i in drive if i.definition.role=='drive_unit']
        a,b=[structural_footprints(i,profile)[1] for i in motors]
        if np.all((a[0]<b[1]+profile.rigid_clearance)&(b[0]<a[1]+profile.rigid_clearance)):
            diagnostics['joint_rejections']['drive_retainer_collision']+=1;continue
        local_drive=_local_halfwidth(volume,wc,radius,bounds)
        if wc[0]+width/2<local_drive-profile.nominal_wall:
            diagnostics['joint_rejections']['drive_track']+=1;continue
        pairs=[]
        for y in ys:
            if (y>wc[1]) != (nearest==0):continue
            wb=abs(y-wc[1])
            if not target['minimum_wheelbase_mm']<=wb<=target['maximum_wheelbase_mm']:
                diagnostics['passive_rejections']['wheelbase']+=1;continue
            if target['measured'] and abs(y-target['target_y_mm'][1-nearest])>target['tolerance_mm']:
                diagnostics['passive_rejections']['source_axle_band']+=1;continue
            source_imbalance=abs((target['target_y_mm'][0]-bounds[0,1])-(bounds[1,1]-target['target_y_mm'][1])) if target['measured'] else 0.
            overhangs=[min(wc[1],y)-bounds[0,1],bounds[1,1]-max(wc[1],y)]
            if abs(overhangs[1]-overhangs[0])>max(target['maximum_overhang_imbalance_mm'],source_imbalance+2*length*policy.source_axle_tolerance_fraction):
                diagnostics['passive_rejections']['overhang_balance']+=1;continue
            z=entry['ground_z']+pr
            local=local_width([wc[0],y,z],pr)
            xs=np.unique(np.concatenate([[wc[0],local],np.arange(max(width,local-width),local+width/2+volume.pitch,volume.pitch*2)]))
            for x in xs:
                if x+width/2<local-profile.nominal_wall:continue
                pair=[placed(passive,np.eye(3),[s*x,y,z],'wheel_passive_'+('left' if s>0 else 'right'),'parallel_axle') for s in [1,-1]]
                exposure=[wheel_side_exposure(volume,p.bounds.mean(0),pr,width,s,samples=96,maximum_recess=profile.nominal_wall) for p,s in zip(pair,[1,-1])]
                if min(e['exposed_fraction'] for e in exposure)<profile.minimum_wheel_exposed_fraction:
                    diagnostics['passive_rejections']['side_exposure']+=1;continue
                axle_y=sorted([wc[1],y]);errors=abs(np.asarray(axle_y)-target['target_y_mm'])
                score=dict(source_axle_error_mm=errors.tolist(),source_correspondence=float(errors.sum()/target['tolerance_mm']),
                    wheelbase_mm=float(wb),wheelbase_fraction=float(wb/length),tire_edge_gap_mm=float(wb-radius-pr),
                    passive_track_fit=float(abs(x-local)/width),drive_track_fit=float(abs(wc[0]-local_drive)/width),
                    passive_appearance_damage=_damage(volume,pair[0].bounds.mean(0),pr,width),
                    drive_appearance_damage=_damage(volume,wc,radius,kit['drive_wheel'].raw['rigid_body']['nominal_width_mm']),
                    mounting_height=float(wc[2]/vehicle.extents[2]))
                source_radii=target.get('source_wheel_radii_mm',[])
                score['tire_proportion_penalty']=float(np.mean(abs(radius/np.asarray(source_radii)-policy.preferred_source_tire_scale))) if source_radii and all(source_radii) else 0.
                total=4*score['source_correspondence']+.8*(score['passive_appearance_damage']+score['drive_appearance_damage'])+.35*(score['passive_track_fit']+score['drive_track_fit'])+.15*score['mounting_height']
                total+=score['tire_proportion_penalty']
                record=dict(entry,joint_axles=True,passive_center=[float(x),float(y),float(z)],axle_y_mm=axle_y,
                    axle_targets=target,score=score,total=float(total),passive_side_exposure=exposure)
                pairs.append((record,drive+pair))
        if not pairs:
            diagnostics['joint_rejections']['no_passive_pair']+=1;continue
        # Retain several different passive tracks/regions for later structure
        # feedback, without flooding the topology search with near duplicates.
        pairs.sort(key=lambda p:p[0]['total']);chosen=[]
        for pair in pairs:
            pc=np.asarray(pair[0]['passive_center'])
            if all(np.linalg.norm(pc-np.asarray(p[0]['passive_center']))>=profile.candidate_pitch for p in chosen):
                chosen.append(pair)
            if len(chosen)>=3:break
        for record,instances in chosen:
            record['index']=len(rows);rows.append(record)
            for inst in instances:inst.score=dict(record['score'])
            accepted.append((record,instances))
    accepted.sort(key=lambda p:p[0]['total'])
    diagnostics.update(joint_candidates=len(rows),accepted=len(rows),diverse=len(accepted))
    return accepted,rows
