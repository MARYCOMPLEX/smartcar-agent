from __future__ import annotations
import numpy as np
from smartcar.geometry.solid import union,box
from smartcar.assembly.motion_validation import linear_path


def plan_assembly(instances,parts,vehicle,profile,records=None):
    steps=[];installed=[]
    offset=np.array([0,0,vehicle.extents[2]+profile.cable_keepout])
    # Open tray assembly: descending pieces with taller/fixed parts first.
    order=sorted([i for i in instances if "wheel" not in i.definition.role],key=lambda i:i.bounds[1,2],reverse=True)
    for inst in order:
        print('ASSEMBLY '+inst.id,flush=True)
        obstacles=[("bottom_cover",parts["bottom_cover"])]+installed
        certificate=linear_path(inst.proxy(),obstacles,offset,profile)
        steps.append(dict(id=inst.id,operation="lower vertically onto open chassis; keep temporarily seated until screw tightening stage",path=certificate))
        installed.append((inst.id,inst.proxy()))
    for name,part in parts.items():
        if "retainer" in name:
            print('ASSEMBLY '+name,flush=True)
            cert=linear_path(part,[("bottom_cover",parts["bottom_cover"])]+installed,offset,profile)
            steps.append(dict(id=name,operation="install motor retainer from above",path=cert))
            installed.append((name,part))
    if records:
        from smartcar.structure.fasteners import head_solid
        for screw in records.get("fasteners",[]):
            if screw["kind"]!="closure":installed.append((screw["id"],head_solid(screw)))
    print('ASSEMBLY body_closure',flush=True)
    body_path=linear_path(parts["body"],[("bottom_cover",parts["bottom_cover"])]+installed,offset,profile)
    steps.append(dict(id="body_closure",operation="lower upper body onto populated chassis; install underside closure screws",path=body_path))
    # Equivalent lid-removal certificate moves the populated tray downward while
    # the upper shell is fixed. This is the opposite relative translation.
    tray=union([parts["bottom_cover"]]+[s for _,s in installed])
    print('ASSEMBLY bottom_lid',flush=True)
    lid_path=linear_path(tray,[("body",parts["body"])],-offset,profile)
    steps.append(dict(id="bottom_lid",operation="remove closure screws and lower populated bottom tray",path=lid_path))
    for inst in instances:
        if "wheel" in inst.definition.role:
            direction=np.array([np.sign(inst.bounds.mean(0)[0]),0,0])*(vehicle.extents[0]/2)
            # A cylinder proxy contains the mating hub's missing void. Mated
            # motor/axle contacts are validated as functional exceptions, not
            # claimed to be exact wheel geometry.
            cert=linear_path(inst.proxy(),[("body",parts["body"]),("bottom_cover",parts["bottom_cover"])],direction,profile)
            steps.append(dict(id=inst.id,operation="slide wheel axially onto shaft/axle; retain with specified fastener",path=cert,
                              interface_status="WARNING: wheel hub CAD/axial detail unavailable"))
    return dict(architecture="removable populated bottom chassis + upper appearance shell",strategy="linear insertion",
                steps=steps,fastener_stage="Tighten internal screws after placing all hardware and retainers, before lowering body. Continuous head/tool sweeps are in validation_report.json.",
                status="FAIL" if any(s["path"]["status"]=="FAIL" for s in steps) else "WARNING" if any(s["path"]["status"]=="WARNING" for s in steps) else "PASS",
                cable_routing="UNVERIFIED: connector frames missing; route and verify before final assembly")
