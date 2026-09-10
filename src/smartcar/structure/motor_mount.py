import numpy as np
from smartcar.geometry.solid import box,cylinder,union
from smartcar.geometry.collision import expand


def motor_mount(inst,floor,profile):
    b=inst.bounds;gap=profile.rigid_clearance;w=profile.support_wall
    inner=expand(b,gap);outer=inner.copy();outer[:,:2]+=np.array([[-w,-w],[w,w]])
    side=np.sign(inst.bounds.mean(0)[0]);outward=1 if side>0 else 0
    # Keep the complete retainer/cradle outside the wheel's axial keepout.
    # A later cylindrical trim can otherwise produce a feathered cap edge.
    limit=np.asarray(inst.metadata["wheel_center"])[0]-side*(inst.metadata["wheel_width_mm"]/2+profile.moving_clearance+profile.mesh_tolerance)
    outer[outward,0]=min(outer[outward,0],limit) if side>0 else max(outer[outward,0],limit)
    outer[0,2]=floor-.2;outer[1,2]=b[0,2]+profile.tray_wall_height
    cut=inner.copy();cut[0,2]=b[0,2];cut[1,2]=outer[1,2]+1
    cradle=box(outer)-box(cut)
    # Rigid supports conform to the authoritative packaging interface. A separate
    # screw bridge retains the gearbox without inventing a motor mounting hole.
    cy=b[:,1].mean();spacing=profile.standoff_radius+profile.support_wall
    strap=2*(spacing+profile.standoff_radius)
    capbounds=np.array([[outer[0,0],cy-strap/2,b[1,2]+gap],[outer[1,0],cy+strap/2,b[1,2]+gap+w]])
    posts=[];holes=[];screw_records=[]
    # Anchor the bridge on the inboard side. Outboard screw posts would compete
    # with the fixed 3.4 mm shaft-to-wheel seating gap and wheel rotation space.
    inward=0 if inst.bounds.mean(0)[0]>0 else 1
    x=b[inward,0]+(-1 if inward==0 else 1)*(gap+profile.standoff_radius)
    capbounds[inward,0]=x+(-1 if inward==0 else 1)*profile.standoff_radius
    cap=box(capbounds)
    for py in [cy-spacing,cy+spacing]:
        rad=profile.standoff_radius
        post=cylinder(rad,floor-.1,b[1,2]+gap,(x,py))
        pilot=cylinder(profile.screw_fit["vertical_pilot"]/2,b[1,2]+gap-profile.screw_fit["thread_length"],b[1,2]+gap+.1,(x,py))
        posts.append(post-pilot)
        holes.append(cylinder(profile.screw_fit["clearance_hole"]/2,capbounds[0,2]-.1,capbounds[1,2]+.1,(x,py)))
        screw_records.append(dict(center_xy=[x,py],cap_top_z=capbounds[1,2],engagement_mm=profile.screw_fit["thread_length"]-w))
    # Prevent post material from entering the true motor envelope. Leaving a
    # concave clipped post is safe only when the independent wall check passes.
    exclusion=expand(b,gap);exclusion[0,2]=b[0,2]
    mounts=union([cradle]+posts)-box(exclusion)
    return mounts,cap-union(holes),dict(hardware=inst.id,cradle_bounds=outer,retainer_bounds=capbounds,screws=screw_records,interface="declared CAD rigid envelope cradle + screw retainer")
