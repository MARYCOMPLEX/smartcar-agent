import numpy as np
from smartcar.geometry.solid import cylinder,union


def pcb_mount(inst,floor,profile):
    hw=inst.definition;r=inst.transform[:3,:3];t=inst.transform[:3,3]
    holes=hw.features.get("mounting_holes",[])
    solids=[];records=[]
    for h in holes:
        axis=r@np.array(h["axis"])
        if abs(axis[2])<.99:continue
        center=r@np.array(h["point"])+t
        # Find CAD board's underside along the insertion direction using its
        # measured pair of large parallel faces, not the overall hardware bounds.
        a=int(np.argmax(abs(np.array(h["axis"]))))
        board_points=[]
        for level in hw.features["support_levels"]:
            point=np.array(h["point"]);point[a]=level;board_points.append(point)
        board_z=[(r@p+t)[2] for p in board_points]
        top=min(board_z)
        rad=profile.standoff_radius
        s=cylinder(rad,floor-.2,top,center[:2])
        hole=cylinder(profile.screw_fit["vertical_pilot"]/2,max(floor-.3,top-profile.screw_fit["thread_length"]),top+.2,center[:2])
        solids.append(s-hole)
        records.append(dict(hardware=inst.id,center_xy=center[:2],support_z=top,board_top_z=max(board_z),radius=rad,pilot_mm=profile.screw_fit["vertical_pilot"],source="CAD cylindrical mounting hole and PCB face"))
    return union(solids),records
