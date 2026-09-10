import numpy as np
from smartcar.geometry.solid import cylinder,intersection_volume,to_mesh
from smartcar.structure.fasteners import head_solid,all_certified_mount_voids


def validate_fasteners(parts,instances,records,profile,ground):
    checks=[];hardware={i.id:all_certified_mount_voids(i) if "wheel" not in i.definition.role else i.proxy() for i in instances}
    def add(name,status,**data):checks.append(dict(check=name,status=status,**data))
    for rec in records.get("fasteners",[]):
        name=rec["id"];head=head_solid(rec);base=np.asarray(rec["head_bearing_center"]);axis=np.asarray(rec["thread_axis"])
        clashes=[];gaps=[]
        for other,solid in {**parts,**hardware}.items():
            v=intersection_volume(head,solid)
            contact=other==rec["owner"]
            required=0. if contact else profile.rigid_clearance
            gap=head.min_gap(solid,2*profile.rigid_clearance) if v<=profile.collision_volume_tolerance else 0.
            gaps.append(dict(obstacle=other,gap_lower_bound_mm=gap,required_mm=required,intersection_mm3=v,contact_allowed=contact))
            if v>profile.collision_volume_tolerance or gap+profile.numerical_tolerance<required:clashes.append(other)
        add("fastener_head:"+name,"FAIL" if clashes else "PASS",measurements=gaps,obstructions=clashes,
            method="circumscribed nominal head envelope; both PCB-side voids independently CAD-certified")
        obstacles={**{k:s for k,s in parts.items() if rec["kind"]=="closure" or k!="body"},**hardware}
        # Exact continuous cylinder sweeps for linear head and tool approach.
        sign=-axis[2];head_top=base[2]+sign*rec["head_height_mm"]
        distant=(max(to_mesh(s).bounds[1,2] for s in parts.values())+profile.tool_radius if sign>0
                 else min(to_mesh(s).bounds[0,2] for s in parts.values())-profile.tool_radius-rec["head_height_mm"])
        head_path=cylinder(rec["head_diameter_mm"]/2/np.cos(np.pi/64),min(base[2],distant),max(base[2],distant),base[:2])
        tool_path=cylinder(profile.tool_radius,min(head_top,distant),max(head_top,distant),base[:2])
        for label,sweep in [("head_insertion",head_path),("tool_access",tool_path)]:
            measured={k:intersection_volume(sweep,s) for k,s in obstacles.items()}
            blocking={k:v for k,v in measured.items() if v>profile.collision_volume_tolerance}
            add(f"fastener_{label}:{name}","FAIL" if blocking else "PASS",maximum_intersection_mm3=max(measured.values(),default=0.),obstructions=blocking,
                method="continuous cylindrical swept volume",assembly_state="body fitted" if rec["kind"]=="closure" else "open populated chassis",tool_radius_mm=profile.tool_radius)
        add("fastener_engagement:"+name,"PASS" if rec["engagement_mm"]>=profile.screw_fit["minimum_engagement"] else "FAIL",
            geometric_engagement_mm=rec["engagement_mm"],minimum_required_mm=profile.screw_fit["minimum_engagement"],thread_retention="requires fit coupon; geometric depth alone does not certify pullout")
        z=to_mesh(head).bounds[0,2]
        add("fastener_ground:"+name,"PASS" if z>=ground+profile.rigid_clearance else "FAIL",minimum_z_mm=z,ground_z_mm=ground)
    return checks
