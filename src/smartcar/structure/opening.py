import numpy as np
from smartcar.geometry.solid import box
from smartcar.geometry.rays import all_ray_hits


def reconstruction_aperture(bounds,profile):
    """Reserve the physical opening plus half-cell-diagonal boundary error."""
    from smartcar.geometry.collision import expand
    uncertainty=.5*np.sqrt(3)*profile.shell_regularization_pitch
    return box(expand(np.asarray(bounds),uncertainty))


def switch_opening(inst,exterior,floor,profile):
    feature=next(f for f in inst.definition.raw["functional_features"] if f.get("external_access_required"))
    b=np.array(feature["local_bounds_xyz_mm"])
    corners=np.array(np.meshgrid(*b.T,indexing="ij")).reshape(3,-1).T
    points=corners@inst.transform[:3,:3].T+inst.transform[:3,3]
    bounds=np.array([points.min(0),points.max(0)])
    center=bounds.mean(0);center[2]=bounds[0,2];axis=np.array([0,0,-1.])
    origins=np.array([center]);dirs=np.array([axis])
    hits,_,_=all_ray_hits(exterior,origins,dirs)
    distances=(hits-center)@axis if len(hits) else np.array([])
    valid=hits[distances>0] if len(hits) else np.empty((0,3))
    if not len(valid):raise ValueError("SWITCH_NO_EXTERIOR_RAY")
    # Derived motion bounds include actuator travel. Finger space is a real
    # aperture enlargement and is tested against mounts and fasteners later.
    opening=bounds.copy();opening[:,:2]+=np.array([[-1,-1],[1,1]])*profile.finger_clearance
    opening[0,2]=min(exterior.bounds[0,2],floor-profile.bottom_thickness)-profile.finger_clearance
    # Stop at the externally facing actuator surface. Cutting to the opposite
    # side of its complete feature box would destroy the PCB mounting columns.
    opening[1,2]=bounds[0,2]+profile.rigid_clearance
    return box(opening),dict(hardware=inst.id,feature=feature["id"],feature_bounds=bounds,opening_bounds=opening,
                            ray_origin=center,ray_direction=axis,exterior_intersections=valid,access_surface="bottom",status="derived and pending final collision validation")
