"""Geometry of certified closure connections shared by synthesis stages."""
import numpy as np
from smartcar.geometry.solid import box, cylinder


def rib_solid(record):
    if record['type']=='vertical':
        return cylinder(record['radius_mm'],record['z0_mm'],record['z1_mm'],record['center_xy'])
    length=record['length_mm'];width=record['width_mm']
    rib=box([[0,-width/2,record['z0_mm']],[length,width/2,record['z1_mm']]])
    return rib.rotate((0,0,record['angle_deg'])).translate((*record['center_xy'],0))
