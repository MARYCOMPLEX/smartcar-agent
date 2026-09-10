import numpy as np
from smartcar.geometry.solid import cylinder,union,intersection_volume
from smartcar.geometry.collision import aabb_distance
from smartcar.structure.closure_bores import closure_pilot_end_z


def screw_closure(body,chassis,instances,vehicle,floor,profile):
    radius=profile.fastener_boss_radius
    choices=[]
    clearance_by_center={}
    rejected_support_clearance=0
    for x in np.arange(vehicle.bounds[0,0]+radius,vehicle.bounds[1,0]-radius,profile.candidate_pitch):
        for y in np.arange(vehicle.bounds[0,1]+radius,vehicle.bounds[1,1]-radius,profile.candidate_pitch):
            z=floor+profile.sliding_clearance
            boss=cylinder(radius,z,z+profile.screw_fit["thread_length"],(x,y))
            if any(aabb_distance(np.array([[x-radius,y-radius,z],[x+radius,y+radius,z+profile.screw_fit["thread_length"]]]),i.bounds)<profile.rigid_clearance for i in instances):continue
            original_volume=boss.volume()
            overlap=intersection_volume(boss,body)
            # Must be already embedded in load-bearing shell material, with a
            # complete outer cylinder (no unsupported, floating screw bosses).
            if overlap<original_volume*.98:continue
            # The manufactured interface restores the complete native cylinder.
            # A small overlap lost to a nearby support pocket can pass the shell
            # embedding test while consuming the actual lid sliding clearance.
            # Check the complete boss against the generated chassis, including
            # its axle brackets, posts, trays and connecting rails.
            support_gap=boss.min_gap(chassis,profile.sliding_clearance*2)
            if support_gap+profile.numerical_tolerance<profile.sliding_clearance:
                rejected_support_clearance+=1
                continue
            below=cylinder(radius,floor-profile.bottom_thickness,floor,(x,y))
            if intersection_volume(below,chassis)<below.volume()*.98:continue
            choices.append((x,y))
            clearance_by_center[(x,y)]=float(support_gap)
    selected=[]
    for target in [(-1,-1),(1,1),(-1,1),(1,-1)]:
        allowed=[p for p in choices if all(np.linalg.norm(np.array(p)-q)>radius*3 for q in selected)]
        if not allowed:break
        point=max(allowed,key=lambda p:p[0]*target[0]+p[1]*target[1])
        selected.append(point)
    body_holes=[];lid_holes=[]
    for xy in selected:
        body_holes.append(cylinder(profile.screw_fit["vertical_pilot"]/2,floor-.1,closure_pilot_end_z(floor,profile),xy))
        lid_holes.append(cylinder(profile.screw_fit["clearance_hole"]/2,floor-profile.bottom_thickness-.1,floor+.1,xy))
    return body-union(body_holes),chassis-union(lid_holes),dict(type="M2.5 screw closure",centers_xy=selected,
                screw_count=len(selected),seam_z=floor,gap_mm=profile.sliding_clearance,
                support_clearance_mm=[clearance_by_center[xy] for xy in selected],
                candidates_rejected_support_clearance=rejected_support_clearance,
                engagement_mm=profile.screw_fit["thread_length"]-profile.bottom_thickness-profile.sliding_clearance,
                head_access="straight from underside; heads protrude by hardware head height",minimum_required_count=4)
