from dataclasses import replace
import copy
import numpy as np
from smartcar.geometry.sdf import DesignableVolume


def adopt_ground(vehicle,volume,instances,floor,ground):
    offset=np.array([0.,0.,-ground]);converted=[]
    for inst in instances:
        mat=inst.transform.copy();mat[:3,3]+=offset
        metadata=copy.deepcopy(inst.metadata)
        for key in ["shaft_tip","wheel_center"]:
            if key in metadata:metadata[key]=np.asarray(metadata[key])+offset
        converted.append(replace(inst,transform=mat,bounds=inst.bounds+offset,metadata=metadata))
    volume2=DesignableVolume(volume.occupancy,volume.origin+offset,volume.pitch,volume.wall)
    return vehicle.copy().apply_translation(offset),volume2,converted,floor-ground
