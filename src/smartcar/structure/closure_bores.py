"""Choose blind or through bores from actual material above the screw interface."""
import numpy as np
from smartcar.geometry.solid import to_mesh,cylinder,union
from smartcar.geometry.rays import all_ray_hits


def closure_pilot_end_z(seam,profile):
    # Closure screws bear on the underside of the bottom plate. Their tip is
    # NOT one whole screw length above the upper shell seam. Match the same
    # bearing datum used by fastener_records, then add the physical tip gap.
    bearing=seam-profile.bottom_thickness
    return bearing+profile.screw_fit['thread_length']+profile.rigid_clearance+profile.numerical_tolerance


def closure_bores(body,closure,profile):
    if not closure['centers_xy']:return union([]),[]
    seam=closure['seam_z'];end=closure_pilot_end_z(seam,profile)
    radius=profile.screw_fit['vertical_pilot']/2
    mesh=to_mesh(body);cutters=[];records=[]
    angles=np.linspace(0,2*np.pi,64,endpoint=False)
    disk=np.vstack([[0,0],np.column_stack([np.cos(angles),np.sin(angles)])*radius*.99])
    for xy in closure['centers_xy']:
        origins=np.column_stack([disk+xy,np.full(len(disk),end)])
        inside=mesh.contains(origins)
        exits=np.full(len(disk),np.inf)
        if inside.any():
            hits,rays,_=all_ray_hits(mesh,origins,np.tile([0,0,1],(len(origins),1)))
            if len(hits):
                height=hits[:,2]-end;valid=height>profile.numerical_tolerance
                np.minimum.at(exits,rays[valid],height[valid])
        remaining=exits[inside&np.isfinite(exits)]
        minimum=float(remaining.min()) if len(remaining) else None
        # A near-surface blind endpoint leaves an unprintable cap. Continue the
        # same hardware-derived pilot axis through the body, without moving the
        # closure or changing the required engagement. Final validators measure
        # the perforated shell and its actual assembly paths again.
        through=minimum is not None and minimum<profile.minimum_wall+np.sqrt(3)*profile.shell_regularization_pitch
        chosen=float(mesh.bounds[1,2]+profile.numerical_tolerance) if through else end
        cutters.append(cylinder(radius,seam-.1,chosen,xy))
        records.append(dict(center_xy=xy,pilot_diameter_mm=2*radius,nominal_end_z_mm=end,
            screw_bearing_z_mm=seam-profile.bottom_thickness,
            screw_tip_z_mm=seam-profile.bottom_thickness+profile.screw_fit['thread_length'],
            required_tip_clearance_mm=profile.rigid_clearance,
            chosen_end_z_mm=chosen,measured_minimum_vertical_cap_mm=minimum,sampled_columns=len(disk),
            type='through' if through else 'blind_or_ends_in_existing_void'))
    return union(cutters),records
