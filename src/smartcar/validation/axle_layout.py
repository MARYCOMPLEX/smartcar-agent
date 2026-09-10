"""Independent checks of actual hardware poses against appearance evidence.

Do not read solver scores or solver PASS flags here. Recompute axle spacing,
matching and overhangs from the four physical wheel bounding boxes.
"""
import numpy as np
from smartcar.domain.vehicle_policy import VehiclePolicy


def validate_axle_positions(centers, radii, source_bounds, source_axles, scale, profile, policy=None):
    policy=policy or VehiclePolicy();centers=np.asarray(centers);radii=np.asarray(radii)
    bounds=np.asarray(source_bounds)*scale;length=np.ptp(bounds,axis=0)[1]
    checks=[]
    def add(name,passed,**values):checks.append(dict(check=name,status='PASS' if passed else 'FAIL',measurements=values))
    if centers.shape!=(4,3):
        add('axle_layout:four_wheels',False,wheel_count=len(centers));return checks
    order=np.argsort(centers[:,1]);rear,front=centers[order[:2]],centers[order[2:]]
    axle_y=np.array([rear[:,1].mean(),front[:,1].mean()]);base=float(np.diff(axle_y)[0])
    pair_error=float(max(np.ptp(rear[:,1]),np.ptp(front[:,1]),np.ptp(centers[:,2])))
    add('axle_layout:paired_axles',pair_error<=profile.numerical_tolerance,error_mm=pair_error,required_mm=profile.numerical_tolerance)
    min_base=max(length*policy.minimum_wheelbase_fraction,2*radii.max()*(1+policy.minimum_tire_gap_diameters))
    source=[a['y_mm']*scale for a in source_axles]
    if len(source)==2:min_base=max(min_base,(max(source)-min(source))*policy.minimum_source_wheelbase_retention)
    add('axle_layout:wheelbase',min_base-profile.numerical_tolerance<=base<=length*policy.maximum_wheelbase_fraction+profile.numerical_tolerance,
        wheelbase_mm=base,body_length_mm=float(length),wheelbase_fraction=base/length,minimum_mm=float(min_base),maximum_mm=float(length*policy.maximum_wheelbase_fraction),
        tire_edge_gap_mm=base-float(radii[order[:2]].max()+radii[order[2:]].max()))
    if len(source)==2:
        error=np.abs(axle_y-np.sort(source));tolerance=max(profile.candidate_pitch,length*policy.source_axle_tolerance_fraction)
        add('axle_layout:source_correspondence',error.max()<=tolerance+profile.numerical_tolerance,
            actual_axle_y_mm=axle_y.tolist(),source_axle_y_mm=sorted(source),errors_mm=error.tolist(),maximum_error_mm=float(error.max()),tolerance_mm=float(tolerance))
        source_radii=[a.get('radius_mm') for a in sorted(source_axles,key=lambda a:a['y_mm'])]
        if all(source_radii):
            ratios=np.array([radii[order[:2]].mean(),radii[order[2:]].mean()])/(np.asarray(source_radii)*scale)
            small=bool(ratios.min()<policy.minimum_source_tire_scale)
            checks.append(dict(check='axle_layout:source_tire_proportion',status=('FAIL' if policy.source_tire_scale_is_hard else 'WARNING') if small else 'PASS',
                measurements=dict(real_to_source_radius_ratio=ratios.tolist(),minimum_reference_ratio=policy.minimum_source_tire_scale),
                reason='appearance tire radii are geometric references; fixed physical tire sizes are unchanged'))
    else:
        checks.append(dict(check='axle_layout:source_correspondence',status='WARNING',reason='two reliable source axles were not measured; explicit generic proportion policy used'))
    overhang=np.array([axle_y[0]-bounds[0,1],bounds[1,1]-axle_y[1]])
    source_imbalance=abs((source[0]-bounds[0,1])-(bounds[1,1]-source[1])) if len(source)==2 else 0.
    allowed=max(length*policy.maximum_overhang_imbalance_fraction,source_imbalance+length*policy.source_axle_tolerance_fraction*2)
    add('axle_layout:overhang_balance',overhang.min()>=0 and abs(np.diff(overhang)[0])<=allowed+profile.numerical_tolerance,
        rear_front_overhang_mm=overhang.tolist(),difference_mm=float(abs(np.diff(overhang)[0])),maximum_difference_mm=float(allowed))
    symmetry=float(max(abs(rear[:,0].sum()),abs(front[:,0].sum())))
    add('axle_layout:bilateral_balance',symmetry<=profile.numerical_tolerance,axle_midplane_error_mm=symmetry/2)
    return checks


def validate_axle_layout(instances,evidence,scale,profile,policy=None,volume=None):
    wheels=[i for i in instances if 'wheel' in i.definition.role]
    checks=validate_axle_positions([i.bounds.mean(0) for i in wheels],
        [i.definition.raw['rigid_body']['nominal_diameter_mm']/2 for i in wheels],
        evidence['source_bounds_mm'],evidence.get('axles',[]),scale,profile,policy)
    if volume is not None:
        from smartcar.geometry.vehicle_section import axle_body_halfwidth
        for inst in wheels:
            c=inst.bounds.mean(0);radius=inst.definition.raw['rigid_body']['nominal_diameter_mm']/2
            width=inst.definition.raw['rigid_body']['nominal_width_mm']
            reference=axle_body_halfwidth(volume,c,radius,np.asarray(evidence['source_bounds_mm'])*scale)
            recess=reference-(abs(c[0])+width/2)
            protrusion=abs(c[0])-reference
            passed=recess<=profile.nominal_wall+profile.numerical_tolerance and protrusion<=width/2+profile.moving_clearance+profile.numerical_tolerance
            checks.append(dict(check='axle_layout:body_track:'+inst.id,status='PASS' if passed else 'FAIL',
                measurements=dict(local_body_halfwidth_mm=reference,wheel_center_lateral_mm=float(abs(c[0])),face_recess_mm=float(recess),
                    maximum_recess_mm=profile.nominal_wall,center_protrusion_mm=float(protrusion),maximum_protrusion_mm=width/2+profile.moving_clearance)))
    return checks
