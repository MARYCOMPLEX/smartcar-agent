"""Independent access rays against the final generated print geometry."""
import numpy as np
import trimesh
from smartcar.geometry.solid import to_mesh
from smartcar.geometry.wheel import wheel_face_points


def validate_wheel_access(parts,instances,profile):
    # Concatenation preserves all possible blockers without using the aperture
    # construction or the layout field as a validation certificate.
    blockers=trimesh.util.concatenate([to_mesh(part) for part in parts.values()])
    checks=[]
    for inst in instances:
        if 'wheel' not in inst.definition.role:continue
        c=inst.bounds.mean(0);side=1 if c[0]>0 else -1
        rigid=inst.definition.raw['rigid_body']
        points=wheel_face_points(c,rigid['nominal_diameter_mm']/2,rigid['nominal_width_mm'],side)
        directions=np.zeros_like(points);directions[:,0]=side
        blocked=blockers.ray.intersects_any(points,directions)
        fraction=float(np.mean(~blocked))
        required=getattr(profile,'minimum_wheel_exposed_fraction',.5)
        checks.append(dict(check='wheel_side_access:'+inst.id,status='PASS' if fraction>=required else 'FAIL',
                           clear_fraction=fraction,required_fraction=required,rays=len(points),blocked_rays=int(blocked.sum()),
                           method='outward rays from area-uniform wheel face samples against every final print part'))
    return checks
