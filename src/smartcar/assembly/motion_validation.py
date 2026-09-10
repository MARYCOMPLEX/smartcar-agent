from __future__ import annotations
import numpy as np
from smartcar.geometry.solid import box,intersection_volume


def linear_path(moving,obstacles,offset,profile):
    """Discrete samples with a Lipschitz gap bound, plus continuous conservative
    AABB sweep. Never calls a sampled collision-free path a continuous proof.
    offset is the removal direction; insertion is the reverse.
    """
    offset=np.asarray(offset,dtype=float)
    # A strictly positive distance bounds a collision-free interval of the path.
    # Adaptive certified steps avoid hundreds of expensive full-body Booleans
    # after the part has already moved away from its closest obstacle.
    length=float(np.linalg.norm(offset));position=0.;adaptive_samples=0;adaptive_max=0.;adaptive_min=1e6;covered=True
    adaptive_failures=[]
    search_distance=max(profile.rigid_clearance*2,length/16)
    while adaptive_samples<2048:
        at=moving.translate(tuple(position*offset));gap=search_distance
        for name,obs in obstacles:
            v=intersection_volume(at,obs);adaptive_max=max(adaptive_max,v)
            if v>profile.collision_volume_tolerance:
                adaptive_failures.append(dict(fraction=position,obstacle=name,intersection_mm3=v));break
            gap=min(gap,at.min_gap(obs,search_distance))
        adaptive_samples+=1;adaptive_min=min(adaptive_min,gap)
        if adaptive_failures:covered=False;break
        if position>=1 or length<profile.numerical_tolerance:break
        safe=gap-profile.numerical_tolerance
        if safe<=profile.numerical_tolerance:covered=False;break
        position=min(1.,position+safe/length)
    else:covered=False
    if covered:
        return dict(status='PASS',removal_offset_mm=offset,samples=adaptive_samples,max_intersection_mm3=adaptive_max,
                    minimum_sampled_gap_mm=adaptive_min if obstacles else None,continuous_certificate=True,
                    continuous_method='adaptive intervals certified by translation distance Lipschitz bound',failures=[])
    if adaptive_failures:
        return dict(status='FAIL',removal_offset_mm=offset,samples=adaptive_samples,max_intersection_mm3=adaptive_max,
                    minimum_sampled_gap_mm=0.,continuous_certificate=False,continuous_method=None,failures=adaptive_failures)
    n=max(2,int(np.ceil(np.linalg.norm(offset)/profile.assembly_step))+1)
    max_overlap=0.;minimum_gap=1e6;failures=[]
    for i,t in enumerate(np.linspace(0,1,n)):
        at=moving.translate(tuple(t*offset))
        for name,obs in obstacles:
            vol=intersection_volume(at,obs)
            max_overlap=max(vol,max_overlap)
            if vol>profile.collision_volume_tolerance:
                failures.append(dict(step=i,fraction=float(t),obstacle=name,intersection_mm3=vol))
                break
            minimum_gap=min(minimum_gap,at.min_gap(obs,profile.assembly_step+profile.rigid_clearance))
        if failures: break
    # A prism enclosing the entire translation is a conservative continuous
    # certificate. False positives are reported separately, not concealed.
    from smartcar.geometry.solid import to_mesh
    b=to_mesh(moving).bounds
    sweep=box(np.array([b[0]+np.minimum(offset,0),b[1]+np.maximum(offset,0)]))
    continuous_max=max((intersection_volume(sweep,o) for _,o in obstacles),default=0.)
    continuous_pass=continuous_max<=profile.collision_volume_tolerance
    step_distance=float(np.linalg.norm(offset)/(n-1))
    # Distance between disjoint solids changes by at most the translation length.
    # Every path point is within half a step of a checked sample.
    lipschitz_bound=minimum_gap-step_distance/2
    method="full enclosing prism" if continuous_pass else None
    if not failures and lipschitz_bound>profile.numerical_tolerance:
        continuous_pass=True;method="sampled distances + rigorous translation Lipschitz bound"
    minkowski_intersection=None
    if not failures and not continuous_pass and moving.num_tri()<=profile.continuous_proxy_face_limit:
        # The leading boundary prisms form the exact polyhedral translation
        # sweep, retaining certified mounting-column cavities.
        from smartcar.assembly.sweep import translation_sweep
        exact_enclosing_sweep=translation_sweep(moving,offset)
        minkowski_intersection=max((intersection_volume(exact_enclosing_sweep,o) for _,o in obstacles),default=0.)
        if minkowski_intersection<=profile.collision_volume_tolerance:
            continuous_pass=True;method="union of leading boundary triangle prisms"
    return dict(status="PASS" if not failures and continuous_pass else ("FAIL" if failures else "WARNING"),
                removal_offset_mm=offset,samples=n,max_intersection_mm3=max_overlap,minimum_sampled_gap_mm=minimum_gap if obstacles else None,
                continuous_enclosing_prism_intersection_mm3=continuous_max,continuous_certificate=continuous_pass,
                continuous_method=method,translation_gap_lower_bound_mm=lipschitz_bound,
                minkowski_sweep_intersection_mm3=minkowski_intersection,failures=failures[:5])
