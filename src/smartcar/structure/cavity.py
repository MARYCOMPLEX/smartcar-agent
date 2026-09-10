import numpy as np
import itertools
from smartcar.geometry.solid import box,union
from smartcar.geometry.collision import expand


def cavities(instances,floor,profile):
    out=[];metadata=[]
    for inst in instances:
        if "wheel" in inst.definition.role:continue
        b=expand(inst.bounds,profile.rigid_clearance+profile.support_wall+profile.sliding_clearance)
        # The top does not need side-wall allowance. The underside is a swept
        # assembly opening, not a uniformly scaled copy of the exterior.
        uncertainty=.5*np.sqrt(3)*profile.shell_regularization_pitch
        b[1,2]=inst.bounds[1,2]+profile.rigid_clearance+uncertainty
        b[0,2]=floor-profile.bottom_thickness*2
        out.append(box(b));metadata.append(dict(hardware=inst.id,bounds=b,method="rigid envelope + mount allowance + linear insertion sweep"))
    return union(out),metadata


def remove_thin_internal_webs(cavity_bounds,allowed_solid,profile):
    bridges=[];records=[]
    for a,b in itertools.combinations(cavity_bounds,2):
        a=np.asarray(a);b=np.asarray(b)
        for axis in range(3):
            gap=max(a[0,axis]-b[1,axis],b[0,axis]-a[1,axis])
            if not (0<gap<profile.minimum_wall+2*profile.mesh_tolerance):continue
            other=[k for k in range(3) if k!=axis]
            lo=np.maximum(a[0],b[0]);hi=np.minimum(a[1],b[1])
            if np.any(hi[other]<=lo[other]):continue
            lo[axis]=min(a[1,axis],b[1,axis])-profile.mesh_tolerance
            hi[axis]=max(a[0,axis],b[0,axis])+profile.mesh_tolerance
            bridge=box([lo,hi])^allowed_solid
            if not bridge.is_empty():bridges.append(bridge);records.append(dict(gap_mm=gap,bounds=[lo,hi],removed_mm3=bridge.volume()))
    return union(bridges),records
