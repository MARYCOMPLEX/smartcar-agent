from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from smartcar.geometry.solid import box,move


@dataclass
class HardwareInstance:
    id: str
    definition: object
    transform: np.ndarray
    selected_orientation: str
    bounds: np.ndarray
    score: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    minimum_clearance_mm: float = 0.
    metadata: dict = field(default_factory=dict)

    def mesh(self): return self.definition.mesh.copy().apply_transform(self.transform)

    def proxy(self):
        if "wheel" in self.definition.role:
            from smartcar.geometry.solid import from_mesh
            return from_mesh(self.mesh())
        result=box(self.bounds)
        if self.definition.role=="drive_unit" and "shaft_tip" in self.metadata:
            from smartcar.geometry.solid import cylinder
            tip=np.asarray(self.metadata["shaft_tip"]);axis=np.asarray(self.metadata["shaft_axis"])
            a=int(np.argmax(abs(axis)));side=1 if axis[a]>0 else 0
            end=self.bounds[side,a];lo,hi=sorted([end,tip[a]])
            interface=next(i for i in self.definition.raw["mechanical_interfaces"] if i["type"]=="rotary_output")
            radial=max(interface["profile_xy_mm"])/2
            result=result+cylinder(radial,lo,hi,tip[[j for j in range(3) if j!=a]],axis=a)
        supports=self.definition.features.get("mount_support_sides",{})
        for sign,certificate in supports.items():
            axis=certificate["normal_axis"]
            if abs(self.transform[2,axis]-int(sign))>.01 or not certificate["valid"]:continue
            from smartcar.geometry.solid import cylinder,move
            for reg in certificate["regions"]:
                p=np.asarray(reg["point"]);xy=p[[i for i in range(3) if i!=axis]]
                low=p[axis];high=p[axis]+reg['height']
                # The certified empty column meets a rigid bounding-box face.
                # Extend only OUTSIDE that box to prevent a floating-point
                # membrane at the open end after a rigid coordinate transform.
                # The internal certified support plane must remain unchanged.
                eps=np.finfo(float).eps*max(1.,float(np.abs(self.bounds).max()))*1024
                local_bounds=self.definition.bounding_box
                if abs(low-local_bounds[0,axis])<=eps:low=local_bounds[0,axis]-eps
                if abs(high-local_bounds[1,axis])<=eps:high=local_bounds[1,axis]+eps
                vacancy=cylinder(reg["radius"],low,high,xy,axis=axis)
                result=result-move(vacancy,self.transform)
        return result

    def to_dict(self):
        return dict(id=self.id,hardware_definition=self.definition.id,role=self.definition.role,
                    pose=dict(position=self.transform[:3,3],rotation=self.transform[:3,:3],matrix=self.transform),
                    selected_orientation=self.selected_orientation,bounding_box=self.bounds,
                    minimum_clearance_mm=self.minimum_clearance_mm,constraints=self.constraints,score=self.score,metadata=self.metadata)


def placed(definition, rotation, center, name, orientation, **metadata):
    rotation=np.asarray(rotation)
    corners=np.array(np.meshgrid(*definition.bounding_box.T,indexing="ij")).reshape(3,-1).T
    v=corners@rotation.T
    t=np.asarray(center)-(v.min(0)+v.max(0))/2
    mat=np.eye(4);mat[:3,:3]=rotation;mat[:3,3]=t
    return HardwareInstance(name,definition,mat,orientation,np.array([v.min(0)+t,v.max(0)+t]),metadata=metadata)
