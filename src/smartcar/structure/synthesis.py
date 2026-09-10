from __future__ import annotations
import numpy as np
from smartcar.geometry.solid import box,cylinder,union,from_mesh,to_mesh
from smartcar.geometry.rays import all_ray_hits
from smartcar.structure.cavity import cavities,remove_thin_internal_webs
from smartcar.structure.motor_mount import motor_mount
from smartcar.structure.pcb_mount import pcb_mount
from smartcar.structure.battery_tray import battery_tray
from smartcar.structure.opening import switch_opening,reconstruction_aperture
from smartcar.structure.closure import screw_closure
from smartcar.structure.wheel_arch import wheel_arch
from smartcar.structure.fasteners import fastener_records,upper_body_head_pockets
from smartcar.structure.lightweight import hollow_body
from smartcar.structure.regularize import regularize_shell
from smartcar.io import write_json


def synthesize(vehicle,instances,floor,profile,run,volume=None,fastener_definition=None):
    exterior=from_mesh(vehicle)
    b=vehicle.bounds.copy()
    above=b.copy();above[0,2]=floor+profile.sliding_clearance;above[1]+=1
    below=b.copy();below[0,2]=floor-profile.bottom_thickness;below[1,2]=floor
    body=exterior^box(above)
    # A constant-thickness extruded section avoids feather edges caused by
    # intersecting a sloping exterior with a horizontal slab.
    section=exterior.slice(floor-profile.bottom_thickness/2).offset(-profile.sliding_clearance)
    section=section.offset(-profile.minimum_wall/2).offset(profile.minimum_wall/2)
    section_components=section.decompose()
    if section_components:section=max(section_components,key=lambda s:s.area())
    chassis=section.extrude(profile.bottom_thickness).translate((0,0,floor-profile.bottom_thickness))
    cavity,crecords=cavities(instances,floor,profile)
    body=body-cavity
    wheel_cuts=[];body_wheel_cuts=[]
    uncertainty=.5*np.sqrt(3)*profile.shell_regularization_pitch
    for inst in instances:
        if "wheel" not in inst.definition.role:continue
        center=inst.bounds.mean(0)
        # Polygonal circular cutters must circumscribe the requested clearance,
        # otherwise chord sag subtracts from the actual minimum radial distance.
        radius=inst.definition.raw["rigid_body"]["nominal_diameter_mm"]/2
        width=inst.definition.raw["rigid_body"]["nominal_width_mm"]
        wheel_cuts.append(wheel_arch(center,radius,width,min(b[0,2],floor-profile.bottom_thickness)-1,profile,exterior_bounds=b))
        body_wheel_cuts.append(wheel_arch(center,radius,width,min(b[0,2],floor-profile.bottom_thickness)-1,profile,extra=uncertainty,exterior_bounds=b))
    cuts=union(wheel_cuts);body=body-union(body_wheel_cuts);chassis=chassis-cuts
    mounts=[];parts={};access_cuts=[];records={"motor":[],"pcb":[],"battery":[],"switch":[],"openings":[]}
    for inst in instances:
        role=inst.definition.role
        if role=="drive_unit":
            mount,cap,rec=motor_mount(inst,floor,profile)
            mounts.append(mount);parts[inst.id+"_retainer"]=cap;records["motor"].append(rec)
            # Output shaft gets a radial clearance channel to the split line.
            tip=np.array(inst.metadata["shaft_tip"]);axis=np.array(inst.metadata["shaft_axis"])
            radius=max(inst.definition.raw["mechanical_interfaces"][0]["profile_xy_mm"])/2+profile.moving_clearance
            xlo,xhi=sorted([inst.bounds[0,0]-profile.nominal_wall*2,inst.bounds[1,0]+profile.nominal_wall*2])
            shaftcut=cylinder(radius,xlo,xhi,tip[1:],axis=0)
            body=body-shaftcut;chassis=chassis-shaftcut
            records["openings"].append(dict(type="shaft",hardware=inst.id,center=tip,axis=axis,radius_mm=radius))
        elif role in ["main_controller","power_switch"]:
            mount,rec=pcb_mount(inst,floor,profile);mounts.append(mount);records["pcb" if role=="main_controller" else "switch"].extend(rec)
        elif role=="battery":
            tray,slots,rec=battery_tray(inst,floor,profile);mounts.append(tray);chassis=chassis-slots;records["battery"].append(rec)
            access_cuts.append(slots)
            # Reconstructing the shell may displace its boundary by half a
            # voxel diagonal. Preserve the physical slot through that process.
            body=body-union([reconstruction_aperture(bound,profile) for bound in rec['strap_passage_bounds']])
            records['openings'].extend(dict(type='battery_strap_channel',hardware=inst.id,opening_bounds=b) for b in rec['strap_passage_bounds'])
    chassis=union([chassis]+mounts)
    # Cut exact functional access only after final pose. Failed opening must
    # trigger layout feedback, not a manually placed visual hole.
    for inst in instances:
        if inst.definition.role=="power_switch":
            opening,orec=switch_opening(inst,vehicle,floor,profile)
            chassis=chassis-opening;body=body-opening;records["openings"].append(orec)
            access_cuts.append(opening)
    # Passive hubs lack authoritative geometry: design bracket with calibrated
    # screw axis, explicitly gate axial engagement until actual hub is measured.
    axles=[];axle_connections=[]
    base_mesh=to_mesh(chassis)
    for inst in instances:
        if inst.definition.role!="passive_wheel":continue
        c=inst.bounds.mean(0);side=np.sign(c[0]);inner=c[0]-side*(inst.definition.raw["rigid_body"]["nominal_width_mm"]/2+profile.moving_clearance)
        thickness=profile.screw_fit["minimum_engagement"]+profile.support_wall
        x0,x1=sorted([inner-side*thickness,inner])
        bracket=box([[x0,c[1]-profile.fastener_boss_radius,floor-.1],[x1,c[1]+profile.fastener_boss_radius,c[2]+profile.fastener_boss_radius]])
        bore=cylinder(profile.screw_fit["horizontal_pilot"]/2,x0-.1,x1+.1,c[1:],axis=0)
        bracket=bracket-bore
        # Connect an axle pedestal to the measured nearest chassis surface along
        # the axle direction; it must not float beside a tapered bottom outline.
        start=np.array([inner-side*thickness/2,c[1],floor-profile.bottom_thickness/2])
        already_attached=bool(base_mesh.contains([start])[0])
        hits,_,_=all_ray_hits(base_mesh,[start],[[-side,0,0]]) if not already_attached else (np.empty((0,3)),None,None)
        distances=(hits-start)@np.array([-side,0,0]) if len(hits) else np.array([])
        hits=hits[distances>=0];distances=distances[distances>=0]
        if len(hits):
            hit=hits[np.argmin(distances)];target=hit[0]-side*profile.nominal_wall
            foot=box([[min(x0,target),c[1]-profile.fastener_boss_radius,floor-profile.bottom_thickness],[max(x1,target),c[1]+profile.fastener_boss_radius,floor]])
            bracket=bracket+foot;axle_connections.append(dict(center=c,attachment_point=hit,bridge_bounds=to_mesh(foot).bounds))
        chassis=chassis+bracket
        axles.append(dict(hardware=inst.id,center=c,bracket_inner_x=inner,pilot_mm=profile.screw_fit["horizontal_pilot"],assumed_inner_hub_contact_x=c[0]-side*inst.definition.raw["rigid_body"]["nominal_width_mm"]/2,
                          warning="Axial wheel web position and screw access missing; bracket is provisional"))
    records["passive_axles"]=axles
    records["axle_connections"]=axle_connections
    records["bottom_section_component_count_before_filter"]=len(section_components)
    # Late passive brackets and their feet must preserve every functional
    # passage, including raised battery straps and the actuator aperture.
    # Otherwise even a small bracket-edge overlap can seal a valid tray slot.
    chassis=chassis-union([cuts]+access_cuts)
    parts={k:s-cuts for k,s in parts.items()}
    from smartcar.structure.bottom_regularization import regularize_bottom
    chassis,records['bottom_regularization']=regularize_bottom(chassis,floor,profile)
    from smartcar.structure.support_connections import connect_tray_supports
    chassis,records['support_connections']=connect_tray_supports(chassis,vehicle,floor,union([cuts]+access_cuts),profile)
    # Upper body must slide past the *generated* support structures as well as
    # hardware. Reserve their vertical assembly projections with lid clearance.
    mount_openings=[]
    for support in mounts+list(parts.values()):
        sb=to_mesh(support).bounds.copy()
        sb[:,:2]+=np.array([[-1,-1],[1,1]])*(profile.sliding_clearance+uncertainty)
        sb[0,2]=floor-profile.bottom_thickness
        sb[1,2]+=profile.sliding_clearance+uncertainty
        mount_openings.append(box(sb))
    for rec in axles:
        c=np.array(rec["center"]);x=rec["bracket_inner_x"]
        sx=profile.screw_fit["minimum_engagement"]+profile.support_wall
        bb=np.array([[min(x,x-np.sign(x)*sx),c[1]-profile.fastener_boss_radius,floor],[max(x,x-np.sign(x)*sx),c[1]+profile.fastener_boss_radius,c[2]+profile.fastener_boss_radius]])
        bb+=np.array([[-1]*3,[1]*3])*(profile.sliding_clearance+uncertainty);bb[0,2]=floor-profile.bottom_thickness
        mount_openings.append(box(bb))
    body=body-union(mount_openings)
    # Reserve installed screw heads as well as the mount solids. These pockets
    # are derived before closure selection so screws cannot be placed in voids.
    records["closure"]=dict(centers_xy=[])
    provisional_fasteners=fastener_records(records,profile,fastener_definition)
    body=body-upper_body_head_pockets(provisional_fasteners,floor,profile)
    if volume is not None:
        allowed=from_mesh(volume.mesh())
        opening_bounds=[c["bounds"] for c in crecords]+[to_mesh(s).bounds for s in mount_openings]
        bridges,bridge_records=remove_thin_internal_webs(opening_bounds,allowed,profile)
        body=body-bridges;records["thin_web_removal"]=bridge_records
    body,chassis,closure=screw_closure(body,chassis,instances,vehicle,floor,profile)
    records["closure"]=closure
    records["fasteners"]=fastener_records(records,profile,fastener_definition)
    if volume is not None:
        body,records["lightweight_shell"]=hollow_body(body,vehicle,volume,closure,profile)
    print("SHELL manufacturing regularization",flush=True)
    restoration_envelope=None
    if volume is not None:
        from smartcar.geometry.repair import occupancy_mesh
        restoration_envelope=from_mesh(occupancy_mesh(volume.distance>=profile.minimum_wall,volume.origin,volume.pitch))
    body,records["shell_regularization"]=regularize_shell(body,closure,profile,restoration_envelope)
    components=body.decompose()
    if len(components)>1:
        components=sorted(components,key=lambda s:s.volume(),reverse=True)
        discarded=components[1:];fraction=sum(s.volume() for s in discarded)/body.volume()
        protected=union([cylinder(profile.fastener_boss_radius,floor,floor+profile.screw_fit["thread_length"],xy) for xy in closure["centers_xy"]])
        intersects_fastener=any((s^protected).volume()>profile.collision_volume_tolerance for s in discarded)
        if fraction<=profile.maximum_unattached_body_fraction and not intersects_fastener:
            records["unattached_construction_fragments_removed"]=[dict(volume_mm3=s.volume(),bounds=to_mesh(s).bounds) for s in discarded]
            body=components[0]
        else:
            records["unattached_fragment_failure"]=dict(fraction=fraction,contains_fastener=intersects_fastener)
    parts={"body":body,"bottom_cover":chassis,**parts}
    from smartcar.geometry.print_mesh import finalize_print_parts
    parts,records['print_mesh_stabilization']=finalize_print_parts(parts,profile)
    for name,s in parts.items():to_mesh(s).export(run/f"09_structure/{name}.stl")
    to_mesh(cavity).export(run/"09_structure/cavity.stl")
    to_mesh(union(mounts)).export(run/"09_structure/mounts.stl")
    to_mesh(cuts).export(run/"05_wheel_candidates/wheel_sweeps.stl")
    write_json(run/"09_structure/structure.json",records)
    write_json(run/"09_structure/cavities.json",crecords)
    write_json(run/"10_openings/openings.json",records["openings"])
    return parts,records
