import numpy as np
from smartcar.geometry.collision import expand


def structural_bounds(inst,profile):
    b=inst.bounds.copy();role=inst.definition.role
    if role=="drive_unit":
        b=expand(b,profile.rigid_clearance+profile.support_wall)
        inward=0 if inst.bounds.mean(0)[0]>0 else 1
        extra=2*profile.standoff_radius-profile.support_wall
        b[inward,0]+=(-extra if inward==0 else extra)
    elif role=="battery":
        b=expand(b,profile.rigid_clearance+profile.support_wall)
        b[1,2]=inst.bounds[1,2]+profile.rigid_clearance
    else:
        b=expand(b,profile.rigid_clearance)
    # Functional apertures consume printable structure as well as empty air.
    # A switch housing can clear a tray while its finger opening cuts the tray
    # wall to a sub-minimum sliver. Reserve transformed actuator travel plus
    # finger access before the solver selects a neighbouring component.
    if role=="power_switch":
        for feature in inst.definition.raw.get('functional_features',[]):
            if not feature.get('external_access_required') or 'local_bounds_xyz_mm' not in feature:continue
            local=np.asarray(feature['local_bounds_xyz_mm'])
            corners=np.array(np.meshgrid(*local.T,indexing='ij')).reshape(3,-1).T
            points=corners@inst.transform[:3,:3].T+inst.transform[:3,3]
            access=np.array([points.min(0),points.max(0)])
            access[:,:2]+=np.array([[-1,-1],[1,1]])*profile.finger_clearance
            b[0,:2]=np.minimum(b[0,:2],access[0,:2]);b[1,:2]=np.maximum(b[1,:2],access[1,:2])
    b[0,2]=inst.bounds[0,2]
    return b


def structural_footprints(inst,profile):
    """Conservative union of actual support regions projected to the tray.
    Two rectangles represent a motor cradle and its localized screw bridge;
    reserving the bridge's maximum width along the full motor wastes real space.
    """
    if inst.definition.role!="drive_unit":return [structural_bounds(inst,profile)[:,:2]]
    b=inst.bounds
    cradle=expand(b,profile.rigid_clearance+profile.support_wall)[:,:2]
    cap=cradle.copy();cy=b[:,1].mean()
    half=2*profile.standoff_radius+profile.support_wall
    cap[:,1]=[cy-half,cy+half]
    inward=0 if b.mean(0)[0]>0 else 1
    cap[inward,0]=b[inward,0]+(-1 if inward==0 else 1)*(profile.rigid_clearance+2*profile.standoff_radius)
    return [cradle,cap]


def footprint_conflict(a,b,profile):
    from smartcar.geometry.collision import overlaps
    for aa in structural_footprints(a,profile):
        for bb in structural_footprints(b,profile):
            if np.all(aa[0]<bb[1]+profile.rigid_clearance) and np.all(bb[0]<aa[1]+profile.rigid_clearance):return True
    return False
