"""Distance-field hollowing with mechanically connected closure material."""
import numpy as np
from smartcar.geometry.solid import box,cylinder,union,from_mesh,to_mesh
from smartcar.structure.closure_ribs import rib_solid
from smartcar.geometry.rays import all_ray_hits


def hollow_body(body,vehicle,volume,closure,profile):
    floor=closure["seam_z"];z0=floor+profile.sliding_clearance
    z1=z0+profile.screw_fit["thread_length"]
    preserved=[];connections=[]
    interior=from_mesh(volume.mesh())
    shell=body-interior
    components=shell.decompose()
    if not components:raise ValueError('CLOSURE_NO_STRUCTURAL_SHELL')
    main=max(components,key=lambda s:s.volume());main_mesh=to_mesh(main)
    # Connect to actual shell remaining AFTER wheel/access/cavity cuts. A ray
    # to the source envelope can terminate in a new opening and leave a floating
    # screw boss. Every proposed clipped rib is checked for solid connectivity.
    for xy in closure["centers_xy"]:
        p=np.array([*xy,(z0+z1)/2]);angles=np.arange(16)*2*np.pi/16
        directions=np.column_stack([np.cos(angles),np.sin(angles),np.zeros(16)])
        origins=np.broadcast_to(p,(16,3))
        locations,rays,_=all_ray_hits(main_mesh,origins,directions)
        lengths=np.linalg.norm(locations-p,axis=1)
        valid=lengths>profile.numerical_tolerance
        boss=cylinder(profile.fastener_boss_radius,z0,z1,xy)
        clipped_boss=boss^body;chosen=None
        # A vertical internal pillar attaches to the upper shell and is less
        # sensitive to a thin lower wheel-arch lip disappearing in manufacture.
        # It is used only when clipping against existing cleared body material
        # preserves a real continuous connection to the structural shell.
        hits,_,_=all_ray_hits(main_mesh,[p],[[0,0,1]])
        hits=hits[hits[:,2]>z1] if len(hits) else hits
        for target in sorted(hits,key=lambda h:h[2]):
            rec=dict(type='vertical',center_xy=list(xy),radius_mm=profile.fastener_boss_radius,z0_mm=z0,z1_mm=float(target[2]+profile.nominal_wall),target_mm=target.tolist())
            rib=rib_solid(rec)^body
            connected=any((c^main).volume()>profile.collision_volume_tolerance and (c^clipped_boss).volume()>=clipped_boss.volume()-profile.collision_volume_tolerance for c in union([main,rib,clipped_boss]).decompose())
            if connected:chosen=(rib,rec);break
        for selected in np.flatnonzero(valid)[np.argsort(lengths[valid])]:
            if chosen is not None:break
            direction=directions[rays[selected]]
            rec=dict(type='horizontal',center_xy=list(xy),length_mm=float(lengths[selected]+profile.nominal_wall),width_mm=profile.nominal_wall,
                     z0_mm=z0,z1_mm=z1,angle_deg=float(np.degrees(np.arctan2(direction[1],direction[0]))),target_mm=locations[selected].tolist())
            rib=rib_solid(rec)^body
            merged=union([main,rib,clipped_boss])
            connected=any((c^main).volume()>profile.collision_volume_tolerance and (c^clipped_boss).volume()>=clipped_boss.volume()-profile.collision_volume_tolerance for c in merged.decompose())
            if connected:chosen=(rib,rec);break
        if chosen is None:
            hits,_,_=all_ray_hits(main_mesh,[p],[[0,0,1]])
            hits=hits[hits[:,2]>z1] if len(hits) else hits
            for target in sorted(hits,key=lambda h:h[2]):
                rec=dict(type='vertical',center_xy=list(xy),radius_mm=profile.fastener_boss_radius,z0_mm=z0,z1_mm=float(target[2]+profile.nominal_wall),target_mm=target.tolist())
                rib=rib_solid(rec)^body
                connected=any((c^main).volume()>profile.collision_volume_tolerance and (c^clipped_boss).volume()>=clipped_boss.volume()-profile.collision_volume_tolerance for c in union([main,rib,clipped_boss]).decompose())
                if connected:chosen=(rib,rec);break
        if chosen is None:raise ValueError('CLOSURE_NO_CONNECTED_RIB')
        rib,rec=chosen;preserved.extend([rib,boss]);connections.append(rec)
    closure['connection_ribs']=connections
    before=body.volume()
    result=body-(interior-union(preserved))
    return result,dict(method="distance-field core; closure ribs connected to actual cut shell by independent Boolean component checks",before_mm3=before,
                       after_mm3=result.volume(),removed_mm3=before-result.volume(),connections=connections)
