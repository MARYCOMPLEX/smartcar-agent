from __future__ import annotations
import numpy as np
from smartcar.geometry.solid import intersection_volume


def aabb_distance(a,b):
    a,b=np.asarray(a),np.asarray(b)
    gap=np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)
    if np.any(gap): return float(np.linalg.norm(gap))
    return -float(np.min(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])))


def overlaps(a,b,clearance=0):
    return bool(np.all(np.asarray(a)[0] < np.asarray(b)[1]+clearance) and np.all(np.asarray(b)[0] < np.asarray(a)[1]+clearance))


def expand(bounds,c):
    return np.asarray(bounds)+np.array([[-c]*3,[c]*3])


def translated_bounds(bounds, offset): return np.asarray(bounds)+np.asarray(offset)


def swept_bounds(bounds,offset):
    b=np.asarray(bounds);o=np.asarray(offset)
    return np.array([b[0]+np.minimum(0,o), b[1]+np.maximum(0,o)])


def pose_clearance(instance,volume,required_wall,clearance):
    b=expand(instance.bounds,clearance)
    # Validate every voxel center in the occupied AABB, plus its boundaries.
    measured=volume.box_clearance(b)
    return measured>=required_wall,measured


def solid_check(a,b,profile,required_gap=0.):
    overlap=intersection_volume(a,b)
    gap=a.min_gap(b,max(required_gap*2,2.0)) if overlap<=profile.collision_volume_tolerance else 0.
    return dict(overlap_mm3=overlap,gap_mm=gap,pass_=overlap<=profile.collision_volume_tolerance and gap+profile.numerical_tolerance>=required_gap)
