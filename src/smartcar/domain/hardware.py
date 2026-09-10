from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import trimesh
import cadquery as cq
import itertools
import hashlib
import json
from OCP.BRepAdaptor import BRepAdaptor_Surface
from smartcar.io import read_json, write_json, sha256


def cq_bounds(shape):
    b=shape.BoundingBox()
    return np.array([[b.xmin,b.ymin,b.zmin],[b.xmax,b.ymax,b.zmax]])


def tessellate(shape,tolerance):
    v,f=shape.tessellate(tolerance, .15)
    mesh=trimesh.Trimesh([x.toTuple() for x in v],f,process=True)
    trimesh.repair.fix_normals(mesh,multibody=True)
    return mesh


def cad_features(shape,hole_diameter=None):
    cylinders=[]; planes=[]
    for face in shape.Faces():
        typ=face.geomType()
        if typ == "CYLINDER":
            c=BRepAdaptor_Surface(face.wrapped).Cylinder()
            p=c.Location(); d=c.Axis().Direction()
            record=dict(radius=c.Radius(),point=[p.X(),p.Y(),p.Z()],axis=[d.X(),d.Y(),d.Z()],bounds=cq_bounds(face),area=face.Area())
            if hole_diameter and abs(2*c.Radius()-hole_diameter)<.02:
                vertex_positions=np.array([v.toTuple() for v in face.Vertices()])
                record["vertex_bounds"]=np.array([vertex_positions.min(0),vertex_positions.max(0)])
            cylinders.append(record)
        elif typ == "PLANE":
            planes.append(dict(center=face.Center().toTuple(),normal=face.normalAt().toTuple(),area=face.Area(),bounds=cq_bounds(face)))
    return dict(cylinders=cylinders,planes=sorted(planes,key=lambda x:x["area"],reverse=True))


def intrinsic_rigid_bounds(raw,features,full_bounds):
    """Match declared rigid dimensions to pairs of large CAD faces.
    STEP cable display poses are not fixed installation poses. Retain their raw
    CAD and flag flexible routing separately; never silently shrink the rigid box.
    """
    nominal=raw.get("rigid_body",{}).get("envelope_xyz_mm")
    if not nominal or raw["role"] not in ["battery","drive_unit"]: return np.array(full_bounds)
    planes=features["planes"]
    options=[]
    for a in planes[:30]:
        ab=np.array(a["bounds"]); spans=np.ptp(ab,axis=0); axis=int(np.argmin(spans))
        for dims in itertools.permutations(nominal):
            other=[i for i in range(3) if i!=axis]
            if max(abs(spans[other]-np.array(dims)[other]))>.05: continue
            for b in planes[:30]:
                bb=np.array(b["bounds"])
                sep=abs(float(np.mean(bb[:,axis])-np.mean(ab[:,axis])))
                if abs(sep-dims[axis])>.05: continue
                bounds=np.array([np.minimum(ab[0],bb[0]),np.maximum(ab[1],bb[1])])
                if max(abs(np.ptp(bounds,axis=0)-dims))<.05:options.append((a["area"]+b["area"],bounds))
    if not options: raise ValueError(f"RIGID_ENVELOPE_NOT_LOCATED:{raw['id']}")
    return max(options,key=lambda x:x[0])[1]


def enrich_intrinsics(hw,profile):
    rigid=intrinsic_rigid_bounds(hw.raw,hw.features,hw.bounding_box)
    hw.features["full_cad_bounds"]=hw.bounding_box.copy()
    hw.features["rigid_bounds"]=rigid
    hw.bounding_box=rigid
    if hw.role in ["battery","drive_unit"]:
        full=np.array(hw.features["full_cad_bounds"])
        hw.features["flexible_geometry_outside_rigid_envelope"]=bool(np.max(abs(full-rigid))>profile.rigid_clearance)
        hw.features["flexible_route_validation"]="UNVERIFIED: declared rigid envelope located on CAD; external cable display pose not constrained"
        # Clip only rendering of the declared rigid part. Exact CAD bytes remain
        # retained and exact verification uses the clipped BRep plus shaft feature.
        b=rigid.copy()
        if hw.role=="drive_unit":
            interface=next(i for i in hw.raw["mechanical_interfaces"] if i["type"]=="rotary_output")
            radius=max(interface["profile_xy_mm"])/2
            c=[c for c in hw.features["cylinders"] if abs(c["radius"]-radius)<.01]
            if not c:raise ValueError("SHAFT_FRAME_NOT_FOUND")
            # An external end of the matched cylinder is the output tip.
            axis=int(np.argmax(abs(np.array(c[0]["axis"]))))
            tip=np.array(max(c,key=lambda c:np.linalg.norm(np.array(c["point"])-rigid.mean(0)))["point"])
            direction=np.zeros(3);direction[axis]=np.sign(tip[axis]-rigid.mean(0)[axis])
            hw.features["shaft"]=dict(tip=tip,axis=direction,engagement=interface["engagement_mm"],source="matched CAD cylindrical surface + supplied interface")
            b[0,axis]=min(b[0,axis],tip[axis]);b[1,axis]=max(b[1,axis],tip[axis])
        crop=cq.Solid.makeBox(*tuple(b[1]-b[0]),cq.Vector(*b[0]))
        hw.exact_geometry=hw.exact_geometry.intersect(crop)
        hw.mesh=tessellate(hw.exact_geometry,profile.mesh_tolerance)
    if hw.role in ["main_controller","power_switch"]:
        diameter=hw.raw.get("mounting_features",{}).get("hole_diameter_mm")
        holes=[]
        for c in hw.features["cylinders"]:
            if diameter and abs(2*c["radius"]-diameter)<.02:
                point=np.array(c["point"])
                if not any(np.linalg.norm(point-np.array(h["point"]))<.01 for h in holes):
                    holes.append(dict(point=point,axis=c["axis"],diameter=diameter,axial_bounds=c.get("vertex_bounds",c["bounds"])))
                else:
                    existing=next(h for h in holes if np.linalg.norm(point-np.array(h["point"]))<.01)
                    bb=np.array(c.get("vertex_bounds",c["bounds"]));old=np.array(existing["axial_bounds"])
                    existing["axial_bounds"]=np.array([np.minimum(old[0],bb[0]),np.maximum(old[1],bb[1])])
        hw.features["mounting_holes"]=holes
        hw.features["board_plane"]=hw.features["planes"][0]
        normal_axis=int(np.argmax(abs(np.array(hw.features["board_plane"]["normal"]))))
        planes=hw.features["planes"][:2]
        # Plated bores can extend beyond the two largest laminate faces. Use
        # exact hole edge endpoints, not inflated tessellation bounding boxes.
        levels=[min(np.array(h["axial_bounds"])[0,normal_axis] for h in holes),max(np.array(h["axial_bounds"])[1,normal_axis] for h in holes)]
        hw.features["support_levels"]=levels
        supports={}
        # Use CAD Boolean checks to establish empty standoff columns. These
        # certified voids can be removed from the conservative AABB proxy.
        cadparts=hw.exact_geometry.Solids()
        if not cadparts:
            cadparts=[cq.Solid.makeSolid(s) for s in hw.exact_geometry.Shells() if s.Closed()]
        truth=cq.Compound.makeCompound(cadparts)
        for sign in [1,-1]:
            lo,hi=(hw.bounding_box[0,normal_axis],levels[0]) if sign==1 else (levels[1],hw.bounding_box[1,normal_axis])
            regs=[]
            for h in holes:
                p=np.array(h["point"]);p[normal_axis]=lo
                direction=np.eye(3)[normal_axis]
                radius=profile.standoff_radius+profile.numerical_tolerance
                probe=cq.Solid.makeCylinder(radius,max(hi-lo,1e-5),cq.Vector(*p),cq.Vector(*direction))
                common=truth.intersect(probe)
                overlap=sum(s.Volume() for s in common.Solids())
                regs.append(dict(point=p,axis=direction,height=hi-lo,radius=radius,cad_intersection_mm3=overlap))
            supports[str(sign)]=dict(valid=bool(regs) and all(r["cad_intersection_mm3"]<=profile.collision_volume_tolerance for r in regs),regions=regs,
                                    method="OCCT intersection of hardware BRep with standoff column",normal_axis=normal_axis)
        hw.features["mount_support_sides"]=supports
    if hw.role=="fastener_reference":
        hw.features["installation_datum"]=fastener_datum(hw.raw,hw.features,hw.mesh)


def fastener_datum(raw,features,mesh):
    """Recover the head bearing plane and direction toward the thread tip.

    A STEP's arbitrary assembly origin is never interpreted as a screw datum.
    Both signs are checked against the supplied head and thread dimensions.
    """
    dimensions=raw["fastener"];radius=dimensions["thread_diameter_mm"]/2
    cylinders=[c for c in features["cylinders"] if abs(c["radius"]-radius)<.02]
    for c in cylinders:
        line=np.asarray(c["point"]);axis=np.asarray(c["axis"])
        for p in features["planes"]:
            n=np.asarray(p["normal"])
            if abs(n@axis)<.999:continue
            datum=line+axis*((np.asarray(p["center"])-line)@n)/(axis@n)
            for sign in [1,-1]:
                direction=axis*sign;z=(mesh.vertices-datum)@direction
                if abs(z.max()-dimensions["thread_length_mm"])<.08 and abs(z.min()+dimensions["head_height_mm"])<.08:
                    return dict(head_bearing_center=datum,thread_axis=direction,
                                measured_head_height_mm=float(-z.min()),measured_thread_length_mm=float(z.max()),
                                source="CAD thread cylinder axis + planar head bearing face; dimensional cross-check")
    raise ValueError("FASTENER_INSTALLATION_DATUM_NOT_FOUND")


@dataclass
class HardwareDefinition:
    id: str
    role: str
    quantity: int | str
    raw: dict
    path: Path
    exact_geometry: object | None
    mesh: trimesh.Trimesh
    bounding_box: np.ndarray
    features: dict
    local_frame: np.ndarray = field(default_factory=lambda:np.eye(4))
    allowed_orientations: list = field(default_factory=list)
    clearance: float = .6

    def summary(self):
        return dict(id=self.id,role=self.role,quantity=self.quantity,bounding_box=self.bounding_box,
                    exact_geometry=str(self.path),collision_geometry="conservative canonical AABB / wheel swept cylinder",
                    local_frame=self.local_frame,allowed_orientations=self.allowed_orientations,clearance=self.clearance,
                    functional_features=self.raw.get("functional_features",[]),mechanical_interfaces=self.raw.get("mechanical_interfaces",[]),
                    keepout_regions=self.raw.get("keepout_regions",[]),assembly_requirements=self.raw.get("service_constraints",{}),
                    features=self.features,authority=self.raw.get("authoritative_status"))


def load_kit(manifest_path, cache_dir, profile):
    manifest_path=Path(manifest_path); manifest=read_json(manifest_path)
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    kit={}
    for item in manifest["components"]:
        p=manifest_path.parent/item["path"]; raw=read_json(p); g=raw["geometry"]
        print(f"CAD {raw['role']}",flush=True)
        geom=g if "path" in g else g["collision_proxy"]
        source=p.parent/geom["path"]
        digest=sha256(source)
        if geom.get("sha256") and digest != geom["sha256"]: raise ValueError(f"HARDWARE_CHECKSUM:{raw['id']}")
        intrinsic_parameters=dict(definition_sha256=sha256(p),mesh_tolerance=profile.mesh_tolerance,
                                  standoff_radius=profile.standoff_radius,numerical_tolerance=profile.numerical_tolerance,
                                  collision_volume_tolerance=profile.collision_volume_tolerance,extractor_version=3)
        intrinsic_key=hashlib.sha256(json.dumps(intrinsic_parameters,sort_keys=True).encode()).hexdigest()[:20]
        intrinsic_base=cache_dir/(digest+'-'+intrinsic_key+'-intrinsic')
        intrinsic_json=intrinsic_base.with_suffix('.json');intrinsic_step=intrinsic_base.with_suffix('.step');intrinsic_mesh=intrinsic_base.with_suffix('.stl')
        if intrinsic_json.exists() and intrinsic_mesh.exists():
            data=read_json(intrinsic_json)
            if not data['has_exact_geometry'] or intrinsic_step.exists():
                shape=cq.importers.importStep(str(intrinsic_step)).val() if data['has_exact_geometry'] else None
                mesh=trimesh.load(intrinsic_mesh,force='mesh')
                kit[raw['role']]=HardwareDefinition(raw['id'],raw['role'],item['quantity'],raw,source,shape,mesh,np.asarray(data['bounding_box']),data['features'],clearance=profile.rigid_clearance)
                print('CAD intrinsic cache hit',flush=True)
                continue
        cached=cache_dir/(digest+".stl"); fcached=cache_dir/(digest+"-features-v2.json")
        shape=None
        if source.suffix.lower() in [".step",".stp"]:
            shape=cq.importers.importStep(str(source)).val()
            bounds=cq_bounds(shape)
            if cached.exists() and fcached.exists():
                mesh=trimesh.load(cached,force="mesh"); features=read_json(fcached)
            else:
                mesh=tessellate(shape,profile.mesh_tolerance); mesh.export(cached)
                features=cad_features(shape,raw.get("mounting_features",{}).get("hole_diameter_mm"));features["solid_count"]=len(shape.Solids());features["cad_valid"]=shape.isValid()
                write_json(fcached,features)
        else:
            mesh=trimesh.load(source,force="mesh"); bounds=mesh.bounds;features={}
        definition=HardwareDefinition(raw["id"],raw["role"],item["quantity"],raw,source,shape,mesh,bounds,features,clearance=profile.rigid_clearance)
        enrich_intrinsics(definition,profile)
        definition.mesh.export(intrinsic_mesh)
        if definition.exact_geometry is not None:cq.exporters.export(definition.exact_geometry,str(intrinsic_step))
        write_json(intrinsic_json,dict(has_exact_geometry=definition.exact_geometry is not None,bounding_box=definition.bounding_box,features=definition.features,parameters=intrinsic_parameters))
        kit[definition.role]=definition
    return kit,manifest
