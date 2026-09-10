from __future__ import annotations
import itertools
import numpy as np
from smartcar.geometry.mesh import mesh_stats
from smartcar.geometry.solid import to_mesh,intersection_volume,from_mesh,box
from smartcar.geometry.collision import aabb_distance
from smartcar.validation.geometry import wall_measurements,overhangs


def validate(parts,instances,records,plan,volume,profile,ground,frame,scale,full=True):
    checks=[]
    def add(name,status,**measurement):checks.append(dict(check=name,status=status,**measurement))
    for name,s in parts.items():
        stabilization=records.get('print_mesh_stabilization',{}).get(name,{})
        add(f"print_mesh_stabilization:{name}",stabilization.get('status','PASS'),measurements=stabilization)
        mesh=to_mesh(s);stats=mesh_stats(mesh)
        valid=stats["watertight"] and stats["manifold"] and stats["winding_consistent"] and stats["component_count"]==1 and stats['degenerate_triangles']==0 and (stats['volume_mm3'] or 0)>0
        add(f"mesh:{name}","PASS" if valid else "FAIL",measurements=stats)
        add(f"self_intersection:{name}","WARNING",method="Manifold Boolean construction enforces oriented manifold output; independent triangle-pair self-intersection audit not implemented")
        if full:
            critical=[]
            if name=='body':
                from smartcar.structure.closure_ribs import rib_solid
                # Independently select final mesh faces around the generated
                # interfaces; changing unrelated geometry cannot hide a small
                # failed junction by moving the random sample sequence.
                for rib in records.get('closure',{}).get('connection_ribs',[]):
                    bounds=to_mesh(rib_solid(rib)).bounds.copy()
                    bounds+=np.array([[-1]*3,[1]*3])*profile.nominal_wall
                    critical.append(bounds)
            thickness=wall_measurements(mesh,profile,regions=critical)
            add(f"wall_thickness:{name}","FAIL" if thickness["below_minimum"] else "WARNING",required_mm=profile.minimum_wall,measurements=thickness)
            add(f"fdm:{name}","WARNING",measurements=overhangs(mesh,profile))
    for a,b in itertools.combinations(parts,2):
        inter=intersection_volume(parts[a],parts[b])
        add(f"print_part_collision:{a}:{b}","PASS" if inter<=profile.collision_volume_tolerance else "FAIL",intersection_mm3=inter)
    for a,b in itertools.combinations(instances,2):
        # Wheel proxy includes its missing shaft bore; a fixed module is allowed
        # to intersect only at that annotated mating interface.
        same=a.metadata.get("module") and a.metadata.get("module")==b.metadata.get("module")
        if same:
            add(f"mating:{a.id}:{b.id}","WARNING",reason="fixed transform constrained; wheel socket geometry unavailable")
            continue
        distance=aabb_distance(a.bounds,b.bounds)
        add(f"hardware_clearance:{a.id}:{b.id}","PASS" if distance>=profile.rigid_clearance else "FAIL",minimum_gap_mm=distance,required_mm=profile.rigid_clearance,method="conservative bounding boxes")
    for inst in instances:
        role=inst.definition.role
        if "wheel" not in role:
            minimum=volume.box_clearance(inst.bounds,step=volume.pitch)
            add(f"containment:{inst.id}","PASS" if minimum>=profile.nominal_wall else "FAIL",distance_to_exterior_mm=minimum,required_mm=profile.nominal_wall,method="complete bounding-volume lattice")
        for name,s in parts.items():
            vol=intersection_volume(inst.proxy(),s)
            allowed_contact=(name=="bottom_cover" and "wheel" not in role) or name.startswith(inst.id+"_retainer")
            gap=inst.proxy().min_gap(s,profile.rigid_clearance*2) if vol<=profile.collision_volume_tolerance else 0
            required=0. if allowed_contact else profile.rigid_clearance
            add(f"hardware_structure:{inst.id}:{name}","PASS" if vol<=profile.collision_volume_tolerance and gap+profile.numerical_tolerance>=required else "FAIL",intersection_mm3=vol,gap_mm=gap,required_mm=required,contact_allowed=allowed_contact,method="conservative proxy (with CAD-certified mount voids) vs final Boolean solid")
        if "wheel" in role:
            center=inst.bounds.mean(0);r=inst.definition.raw["rigid_body"]["nominal_diameter_mm"]/2
            delta=center[2]-r-ground
            add(f"wheel_ground:{inst.id}","PASS" if abs(delta)<=profile.numerical_tolerance else "FAIL",error_mm=delta)
            for name in ["body","bottom_cover"]:
                gap=inst.proxy().min_gap(parts[name],profile.moving_clearance*2)
                add(f"wheel_rotation:{inst.id}:{name}","PASS" if gap+profile.numerical_tolerance>=profile.moving_clearance else "FAIL",minimum_swept_gap_mm=gap,required_mm=profile.moving_clearance,method="full swept nominal cylinder")
        if role=="drive_unit":
            axis=np.asarray(inst.metadata["shaft_axis"]);tip=np.asarray(inst.metadata["shaft_tip"]);wc=np.asarray(inst.metadata["wheel_center"])
            error=np.linalg.norm(np.cross(wc-tip,axis))
            add(f"shaft_alignment:{inst.id}","PASS" if error<profile.numerical_tolerance else "FAIL",radial_error_mm=float(error),angular_error_deg=float(np.degrees(np.arccos(np.clip(abs(axis[0]),0,1)))))
    pcb=[i for i in instances if i.definition.role=="main_controller"][0]
    expected=len(pcb.definition.features.get("mounting_holes",[]))
    add("PCB_mounting","PASS" if len(records["pcb"])==expected and expected>0 else "FAIL",detected_holes=expected,generated_standoffs=len(records["pcb"]),locations=records["pcb"])
    add("battery_mounting","WARNING" if records["battery"] else "FAIL",geometry=records["battery"],reason="tray generated; strap is an additional physical BOM item and load retention requires a bench test")
    from smartcar.validation.battery_access import validate_battery_passages
    checks.extend(validate_battery_passages(parts,records['battery'],profile))
    add('motor_retention','WARNING',maximum_nominal_play_mm=profile.rigid_clearance,reason='CAD rigid envelope cradle and removable retainer generated; exact contact preload, gearbox torque restraint and retention strength are not certified')
    op=next((o for o in records["openings"] if o.get("feature")=="actuator"),None)
    if op:
        access=box(op["opening_bounds"])
        collision=intersection_volume(access,parts["bottom_cover"])
        add("switch_accessibility","PASS" if collision<profile.collision_volume_tolerance else "FAIL",intersection_mm3=collision,opening=op)
    else:add("switch_accessibility","FAIL",reason="no opening")
    closure=records["closure"]
    add("closure_fit","PASS" if closure["screw_count"]>=4 and closure["engagement_mm"]>=profile.screw_fit["minimum_engagement"] else "FAIL",measurements=closure)
    add("closure_intersection","PASS" if intersection_volume(parts["body"],parts["bottom_cover"])<profile.collision_volume_tolerance else "FAIL",intersection_mm3=intersection_volume(parts["body"],parts["bottom_cover"]))
    closure_gap=parts["body"].min_gap(parts["bottom_cover"],profile.sliding_clearance*2)
    add("closure_clearance","PASS" if closure_gap+profile.numerical_tolerance>=profile.sliding_clearance else "FAIL",minimum_gap_lower_bound_mm=closure_gap,required_mm=profile.sliding_clearance)
    from smartcar.validation.fasteners import validate_fasteners
    checks.extend(validate_fasteners(parts,instances,records,profile,ground))
    from smartcar.validation.wheel_access import validate_wheel_access
    checks.extend(validate_wheel_access(parts,instances,profile))
    add("wheel_fastener_accessibility","WARNING",reason="closure/PCB/motor head and tool sweeps checked numerically; wheel axial fastener seats remain unknown")
    for s in plan["steps"]:add("assembly_path:"+s["id"],s["path"]["status"],measurements=s["path"])
    add("connector_accessibility","WARNING",reason="controller connector semantic frames missing; external ports cannot be certified")
    add("flexible_cables","WARNING",reason="full CAD retained; rigid envelopes separated using hardware dimensions; cable routes, connector access and bend radii unverified")
    add("passive_wheel_mounting","WARNING",geometry=records["passive_axles"],reason="hub axial seating/web location missing; axle screw engagement cannot be certified")
    add("fastener_retention","WARNING",pilot_diameter_mm=profile.screw_fit["vertical_pilot"],nominal_thread_diameter_mm=profile.screw_fit["thread_diameter"],reason="supplied empirical vertical pilot is larger than nominal thread; physical coupon required")
    add("coordinate_semantics","WARNING",reason=frame.get("semantic_confidence","geometry-inferred axes"))
    failures=[c for c in checks if c["status"]=="FAIL"]
    return dict(schema_version="validation.v1",status="FAIL" if failures else "WARNING",release_ready=False,
                scale=scale,counts={s:sum(c["status"]==s for c in checks) for s in ["PASS","FAIL","WARNING"]},
                checks=checks,blocking_checks=[c["check"] for c in failures],
                release_statement="Engineering prototype. Unknown wheel hubs, connector semantics, flexible routing and physical retention prevent a production-ready claim.")
