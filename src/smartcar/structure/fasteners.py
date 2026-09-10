"""Geometry-derived screw poses and head keepouts; no vehicle coordinates."""
import numpy as np
import trimesh
from smartcar.geometry.solid import cylinder,union,to_mesh,move


def fastener_records(records,profile,definition=None):
    result=[]
    def add(name,kind,xy,z,sign,owner,engagement):
        axis=np.array([0.,0.,sign]);base=np.array([*xy,z])
        record=dict(id=name,kind=kind,head_bearing_center=base,thread_axis=axis,
                    owner=owner,engagement_mm=engagement,
                    head_diameter_mm=profile.screw_fit["head_diameter"],head_height_mm=profile.screw_fit["head_height"],
                    thread_length_mm=profile.screw_fit["thread_length"],thread_diameter_mm=profile.screw_fit["thread_diameter"])
        if definition is not None:
            datum=definition.features["installation_datum"]
            transform=trimesh.geometry.align_vectors(datum["thread_axis"],axis)
            transform[:3,3]=base-transform[:3,:3]@np.asarray(datum["head_bearing_center"])
            record["pose_matrix"]=transform
            record["hardware_definition"]=definition.id
        result.append(record)
    closure=records["closure"]
    for n,xy in enumerate(closure["centers_xy"]):
        add(f"closure_screw_{n+1}","closure",xy,closure["seam_z"]-profile.bottom_thickness,1,"bottom_cover",closure["engagement_mm"])
    for rec in records["pcb"]+records["switch"]:
        add(f"{rec['hardware']}_screw_{len(result)+1}","PCB",rec["center_xy"],rec["board_top_z"],-1,rec["hardware"],profile.screw_fit["thread_length"]-(rec["board_top_z"]-rec["support_z"]))
    for rec in records["motor"]:
        for n,screw in enumerate(rec["screws"]):
            add(f"{rec['hardware']}_retainer_screw_{n+1}","motor_retainer",screw["center_xy"],screw["cap_top_z"],-1,rec["hardware"]+"_retainer",screw["engagement_mm"])
    return result


def head_solid(record,clearance=0.):
    base=np.asarray(record["head_bearing_center"]);axis=np.asarray(record["thread_axis"])
    end=base-axis*(record["head_height_mm"]+clearance)
    # Circumscribed cylinder bounds the exact button-head envelope.
    radius=(record["head_diameter_mm"]/2+clearance)/np.cos(np.pi/64)
    return cylinder(radius,min(base[2],end[2]),max(base[2],end[2]),base[:2])


def upper_body_head_pockets(records,floor,profile):
    pockets=[]
    for record in records:
        if record["kind"]=="closure":continue
        base=np.asarray(record["head_bearing_center"])
        uncertainty=.5*np.sqrt(3)*profile.shell_regularization_pitch
        radius=(record["head_diameter_mm"]/2+profile.rigid_clearance+profile.mesh_tolerance+uncertainty)/np.cos(np.pi/64)
        top=base[2]+record["head_height_mm"]+profile.rigid_clearance+profile.mesh_tolerance+uncertainty
        pockets.append(cylinder(radius,floor-profile.bottom_thickness,top,base[:2]))
    return union(pockets)


def all_certified_mount_voids(inst):
    """Proxy for head clearance includes independently checked free space on both PCB faces."""
    from smartcar.geometry.solid import box
    proxy=box(inst.bounds)
    for cert in inst.definition.features.get("mount_support_sides",{}).values():
        if not cert["valid"]:continue
        a=cert["normal_axis"]
        for reg in cert["regions"]:
            p=np.asarray(reg["point"]);xy=p[[j for j in range(3) if j!=a]]
            empty=cylinder(reg["radius"],p[a],p[a]+reg["height"],xy,axis=a)
            proxy=proxy-move(empty,inst.transform)
    return proxy
