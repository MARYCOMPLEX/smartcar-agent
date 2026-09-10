import numpy as np
from smartcar.geometry.solid import box,union
from smartcar.geometry.collision import expand


def battery_tray(inst,floor,profile):
    inner=expand(inst.bounds,profile.rigid_clearance)
    ztop=inst.bounds[0,2]+profile.tray_wall_height
    outer=inner.copy();outer[:,:2]+=np.array([[-profile.support_wall]*2,[profile.support_wall]*2])
    outer[0,2]=floor-.2;outer[1,2]=ztop
    cut=inner.copy();cut[0,2]=inst.bounds[0,2];cut[1,2]=ztop+1
    tray=box(outer)-box(cut)
    # Two strap passages. Strap is a BOM item, not an unverified printed snap.
    c=inst.bounds.mean(0);length_axis=int(np.argmax(np.ptp(inst.bounds,axis=0)[:2]));across=1-length_axis
    slots=[];slot_bounds=[]
    for side in [-1,1]:
        p=c.copy();p[across]=inner[0 if side<0 else 1,across]+side*profile.support_wall/2
        b=np.array([p,p]);b[:,length_axis]+=np.array([-1,1])*(profile.battery_strap_width/2+profile.sliding_clearance)
        b[:,across]+=np.array([-1,1])*profile.support_wall
        # Open the slot through the actual raised tray, not only through a
        # nominal bottom channel. An intermediate tray height otherwise leaves
        # an arbitrarily thin roof over the channel (e.g. 1.1 mm).
        b[:,2]=[floor-profile.bottom_thickness-1,max(ztop+profile.numerical_tolerance,floor+profile.battery_strap_thickness+profile.sliding_clearance)]
        slots.append(box(b));slot_bounds.append(b)
    passages=union(slots)
    # The mount's own platform must preserve its strap passages. Otherwise the
    # subsequent chassis/mount union closes a previously drilled slot with a
    # thin floor membrane, preventing threading and violating minimum walls.
    tray=tray-passages
    return tray,passages,dict(hardware=inst.id,inner_bounds=inner,outer_bounds=outer,wall_height=profile.tray_wall_height,
                                strap_passage_bounds=slot_bounds,strap_sliding_clearance_mm=profile.sliding_clearance,
                                retention="removable 4 mm reusable strap through two chassis slots",strap_required=True)
