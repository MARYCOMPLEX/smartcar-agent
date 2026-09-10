import numpy as np
import trimesh
from smartcar.geometry.solid import to_mesh


def measured_surface_deviation(reference,result,profile,minimum_z=None,samples=1600):
    points,_=trimesh.sample.sample_surface(reference,samples,seed=profile.seed)
    if minimum_z is not None:points=points[points[:,2]>minimum_z]
    distances=[]
    for part in np.array_split(points,max(1,int(np.ceil(len(points)/100)))):
        if not len(part):continue
        _,d,_=trimesh.proximity.closest_point(result,part);distances.extend(d)
    d=np.asarray(distances)
    return dict(sample_count=len(d),median_mm=float(np.median(d)),p95_mm=float(np.quantile(d,.95)),maximum_sampled_mm=float(d.max()),
                fraction_within_nominal_wall=float(np.mean(d<=profile.nominal_wall)),method="seeded surface samples to closest triangles; intentional openings included")
