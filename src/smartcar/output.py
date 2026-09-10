from __future__ import annotations
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import trimesh
from smartcar.geometry.solid import to_mesh
from smartcar.io import write_json,read_json
from smartcar.render import render_meshes


def export_3mf(path,parts):
    ns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ET.register_namespace("",ns)
    def tag(n):return "{"+ns+"}"+n
    model=ET.Element(tag("model"),unit="millimeter",attrib={"xml:lang":"en-US"})
    resources=ET.SubElement(model,tag("resources"));build=ET.SubElement(model,tag("build"))
    cursor=0.
    for i,(name,solid) in enumerate(parts.items(),1):
        m=to_mesh(solid); m.apply_translation(-m.bounds[0]);m.apply_translation([cursor,0,0]);cursor+=m.extents[0]+10
        obj=ET.SubElement(resources,tag("object"),id=str(i),type="model",name=name)
        mesh=ET.SubElement(obj,tag("mesh"));vs=ET.SubElement(mesh,tag("vertices"));fs=ET.SubElement(mesh,tag("triangles"))
        for v in m.vertices:ET.SubElement(vs,tag("vertex"),x=f"{v[0]:.6f}",y=f"{v[1]:.6f}",z=f"{v[2]:.6f}")
        for f in m.faces:ET.SubElement(fs,tag("triangle"),v1=str(f[0]),v2=str(f[1]),v3=str(f[2]))
        ET.SubElement(build,tag("item"),objectid=str(i))
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("3D/3dmodel.model",ET.tostring(model,encoding="utf-8",xml_declaration=True))
        z.writestr("[Content_Types].xml",'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def export_output(run,parts,instances,plan,report,records,manifest,profile,scale,ground,attempts,kit=None):
    out=run/"output"
    preparation_path=run/'01_input/input_preparation.json'
    input_preparation=read_json(preparation_path) if preparation_path.exists() else None
    from smartcar.validation.visualize import collision_visualization
    collision_visualization(run,parts,instances,profile)
    meshes=[];exploded=[]
    colors={"body":[161,197,214,130],"bottom_cover":[80,96,121,255],"battery":[240,179,60,255],"main_controller":[61,173,121,255],"power_switch":[176,110,203,255],"drive_unit":[206,101,76,255]}
    for name,s in parts.items():
        m=to_mesh(s);m.export(out/(name+".stl"));m.visual.face_colors=colors.get(name,[175,160,128,255])
        meshes.append((name,m))
        offset=np.array([0,0,35 if name=="body" else -25 if name=="bottom_cover" else 18])
        exploded.append((name,m.copy().apply_translation(offset)))
    from smartcar.validation.exports import validate_stl_exports
    verified=validate_stl_exports(report,out,parts)
    report.clear();report.update(verified)
    write_json(run/"12_validation/validation_report.json",report)
    for i,inst in enumerate(instances):
        m=inst.mesh();m.visual.face_colors=colors.get(inst.definition.role,[42,47,56,255]);meshes.append((inst.id,m))
        delta=[np.sign(inst.bounds.mean(0)[0])*22,0,8] if "wheel" in inst.definition.role else [0,0,12+i*4]
        exploded.append((inst.id,m.copy().apply_translation(delta)))
    fastener=kit.get("fastener_reference") if kit else None
    for screw in records.get("fasteners",[]):
        if fastener is None or "pose_matrix" not in screw:continue
        m=fastener.mesh.copy().apply_transform(screw["pose_matrix"]);m.visual.face_colors=[170,178,188,255]
        meshes.append((screw["id"],m));exploded.append((screw["id"],m.copy().apply_translation(np.asarray(screw["thread_axis"])*-35)))
    for filename,items in [("assembly.glb",meshes),("exploded_assembly.glb",exploded)]:
        scene=trimesh.Scene()
        # glTF is meters. All analysis, STL, layout and reports stay millimeters.
        for name,m in items:scene.add_geometry(m.copy().apply_scale(.001),node_name=name,geom_name=name)
        scene.export(out/filename)
    render_meshes(out/"assembly.png",meshes,"Hardware-aware design - engineering prototype",transparent=True)
    render_meshes(out/"exploded_assembly.png",exploded,"Exploded assembly - geometry-derived placement")
    render_meshes(run/"08_layout/hardware_layout.png",[(i.id,i.mesh()) for i in instances],"Solved hardware poses")
    render_meshes(run/"09_structure/structure.png",[(n,to_mesh(s)) for n,s in parts.items()],"Generated print parts")
    drive_scene=[(i.id,i.mesh()) for i in instances if i.definition.role in ['drive_unit','drive_wheel','passive_wheel']]
    render_meshes(run/"06_wheel_solution/selected.png",drive_scene,"Selected fixed drive modules and passive wheels")
    for name in ['cavity','mounts']:
        mesh=trimesh.load(run/f'09_structure/{name}.stl',force='mesh')
        render_meshes(run/f'09_structure/{name}.png',[(name,mesh)],name.title()+" geometry")
    opening_meshes=[]
    from smartcar.geometry.solid import box,cylinder
    for n,opening in enumerate(records['openings']):
        if 'opening_bounds' in opening:opening_meshes.append((f'collision_opening_{n}',to_mesh(box(opening['opening_bounds']))))
        elif opening.get('type')=='shaft':
            c=np.asarray(opening['center']);opening_meshes.append((f'collision_shaft_{n}',to_mesh(cylinder(opening['radius_mm'],c[0]-profile.nominal_wall,c[0]+profile.nominal_wall,c[1:],axis=0))))
    if opening_meshes:render_meshes(run/'10_openings/openings.png',[(n,to_mesh(s)) for n,s in parts.items()]+opening_meshes,'Computed functional aperture volumes',transparent=True)
    export_3mf(out/"printable.3mf",parts)
    from smartcar.structure.fit_coupon import generate_fit_coupon
    coupon,coupon_record=generate_fit_coupon(profile)
    to_mesh(coupon).export(out/"fastener_fit_coupon.stl")
    write_json(out/"fastener_fit_coupon.json",coupon_record)
    write_json(out/"layout.json",dict(schema_version="layout.v1",unit="mm",scale=scale,input_preparation=input_preparation,ground_z_mm=ground,instances=[i.to_dict() for i in instances]))
    write_json(out/"assembly_plan.json",plan);write_json(out/"validation_report.json",report)
    write_json(out/"fasteners.json",dict(instances=records.get("fasteners",[]),unplaced="wheel axial fasteners: hub seating geometry missing"))
    screws=records["closure"]["screw_count"]+len(records["pcb"])+len(records["switch"])+2*len(records["motor"])+len(records["passive_axles"])
    bom=dict(hardware=[dict(id=x["id"],quantity=x["quantity"]) for x in manifest["components"]],print_parts=list(parts),
             required_m25x6_screws=screws,screw_inventory="physical quantity not provided",additional_items=[dict(item="reusable battery strap",width_mm=profile.battery_strap_width,quantity=1)],
             provisional=["passive axle screws: actual hub section/engagement must be measured"])
    write_json(out/"bom.json",bom)
    write_json(out/"design_metrics.json",dict(scale=scale,part_volumes_mm3={name:solid.volume() for name,solid in parts.items()},
                                             lightweight_shell=records.get('lightweight_shell'),shell_regularization=records.get('shell_regularization')))
    # STEP engineering subassembly retains authoritative BRep hardware. Wheels
    # and appearance-derived print meshes are explicitly absent from this STEP.
    import cadquery as cq
    from scipy.spatial.transform import Rotation
    cad=cq.Assembly(name="solved_rigid_hardware_only")
    for inst in instances:
        shape=inst.definition.exact_geometry
        if shape is not None:
            rotvec=Rotation.from_matrix(inst.transform[:3,:3]).as_rotvec();angle=np.linalg.norm(rotvec)
            axis=rotvec/angle if angle>1e-12 else np.array([0,0,1])
            location=cq.Location(cq.Vector(*inst.transform[:3,3]),cq.Vector(*axis),float(np.degrees(angle)))
            cad.add(shape.moved(location),name=inst.id)
    if fastener is not None:
        for screw in records.get("fasteners",[]):
            if "pose_matrix" not in screw:continue
            mat=np.asarray(screw["pose_matrix"]);rotvec=Rotation.from_matrix(mat[:3,:3]).as_rotvec();angle=np.linalg.norm(rotvec)
            axis=rotvec/angle if angle>1e-12 else np.array([0,0,1])
            location=cq.Location(cq.Vector(*mat[:3,3]),cq.Vector(*axis),float(np.degrees(angle)))
            cad.add(fastener.exact_geometry.moved(location),name=screw["id"])
    cad.export(str(out/"rigid_hardware_assembly.step"))
    lines=["# Hardware-aware smartcar design report", "",f"Validation: **{report['status']}**. Release ready: **NO**.","",
           f"Appearance scale: {scale:.4f}×. {len(instances)} hardware instances. Search attempts: {len(attempts)}.",
           "All dimensions are millimeters; GLB geometry is converted to meters as required by glTF.",
           (f"Input calibration: {input_preparation['basis']}; source coordinate to mm factor {input_preparation['source_coordinate_to_mm_factor']:.9g}; starting vehicle length {input_preparation['prepared_dimensions_mm'][1]:.3f} mm. The appearance scale above is applied after this calibration." if input_preparation else "Input calibration: legacy assumed millimeters."),"",
           "## What was actually calculated","", "- Automatic OBB/PCA vehicle frame with ground-surface evidence; front/rear remains a semantic hypothesis.",
           "- Repaired voxel exterior and conservative signed-distance design space; original source bytes retained.",
           "- CAD-derived output shaft, rigid hardware envelopes, PCB mounting holes and support planes.",
           "- Fixed bilateral motor/wheel modules; candidate erosion and CP-SAT interior layout with hard exclusions.",
           "- Hardware and assembly-driven cavities, bottom tray, mounts, actuator aperture, wheel sweeps and screw closure.",
           "- Independent Boolean collision, clearance, ray thickness sampling and sampled insertion paths.",
           "- Exact-CAD screw placements, conservative head pockets and continuous cylindrical head/tool approach sweeps.","",
           "## Acceptance results","",f"PASS {report['counts']['PASS']}; FAIL {report['counts']['FAIL']}; WARNING {report['counts']['WARNING']}.","",
           "| Check | Result | Evidence |","|---|---|---|"]
    for c in report["checks"]:
        if c["status"]!="PASS":lines.append(f"| {c['check']} | {c['status']} | {c.get('reason',c.get('method','see validation_report.json for measured values'))} |")
    lines += ["","## Fabrication limits","", "Do not treat the preview or manifold STL as a validated working car. FAIL items block fabrication release; WARNING items require annotation, physical measurement, a fit coupon or stronger verification.",
              "The wheel cylinders are collision proxies for supplied physical wheels, never manufacturing wheel geometry. Axial socket seating is an explicit assumption. Passive hub web position is unknown.",
              "The supplied STEP cable display poses are preserved in input data, but flexible cable routing is not certified. Controller connector functions and directions are absent.",
              "Vertical pilot diameter uses the supplied 2.7 mm empirical calibration for a nominal M2.5 screw. Geometric thread retention is not established on an uncalibrated printer.",
              "Ray wall-thickness results are samples, not an exhaustive lower-bound proof. FDM support locations need slicer verification. 3MF parts are arranged for inspection, with no printer build-volume or support profile selected.",
              "","## Scaling interpretation","", "The reported scale is the smallest successful scale among tested strategies and a bounded search; it is not a mathematical global minimum across all possible enclosure architectures.",
              "","## Sources used for implementation","", "- [CadQuery STEP import/export](https://cadquery.readthedocs.io/en/latest/importexport.html)",
              "- [Trimesh voxel creation](https://trimesh.org/trimesh.voxel.creation.html)", "- [Manifold solid operations](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html)"]
    (out/"design_report.md").write_text("\n".join(lines),encoding="utf-8")
